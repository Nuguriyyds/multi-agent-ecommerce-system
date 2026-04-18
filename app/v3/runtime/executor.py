from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from app.v3.hardening import HardeningGate
from app.v3.hooks import HookBus
from app.v3.models import (
    AgentDecision,
    AskClarificationAction,
    CallSubAgentAction,
    CallToolAction,
    FallbackAction,
    HookEvent,
    HookPoint,
    InvocationRecord,
    LoopState,
    Observation,
    PermissionPolicy,
    ReplyToUserAction,
    SessionState,
    SpecialistBrief,
    TaskRecord,
    TaskStatus,
    TraceRecord,
    TurnResult,
    TurnRuntimeContext,
    TurnTask,
    TurnTaskBoard,
)
from app.v3.observability import log_event
from app.v3.registry import CapabilityRegistry, SubAgentProvider, ToolProvider

from .context_packet import ContextPacketBuilder
from .trace_store import TraceStore

DecisionProvider = Callable[[TurnRuntimeContext], Awaitable[AgentDecision]]


class SerialExecutor:
    def __init__(
        self,
        *,
        decision_provider: DecisionProvider,
        registry: CapabilityRegistry | None = None,
        hardening_gate: HardeningGate | None = None,
        trace_store: TraceStore | None = None,
        context_packet_builder: ContextPacketBuilder | None = None,
        hook_bus: HookBus | None = None,
        permission_policy: PermissionPolicy | None = None,
        initial_node: str = "candidate_search",
        max_steps: int = 8,
    ) -> None:
        self._decision_provider = decision_provider
        self._registry = registry or CapabilityRegistry()
        self._hardening_gate = hardening_gate or HardeningGate()
        self._trace_store = trace_store or TraceStore()
        self._context_packet_builder = context_packet_builder or ContextPacketBuilder()
        self._hook_bus = hook_bus
        self._permission_policy = permission_policy
        self._initial_node = initial_node
        self._max_steps = max_steps
        self._logger = logging.getLogger(__name__)

    @property
    def trace_store(self) -> TraceStore:
        return self._trace_store

    async def run_turn(self, session: SessionState, user_message: str) -> TurnResult:
        turn_number = session.turn_count + 1
        trace = TraceRecord(
            trace_id=f"trace-{uuid4().hex[:12]}",
            session_id=session.session_id,
            turn_number=turn_number,
            memory_reads=self._memory_read_keys(session),
        )
        task_board = TurnTaskBoard.create()
        context = TurnRuntimeContext(
            session=session,
            loop_state=LoopState(
                step_number=0,
                current_node=self._initial_node,
                current_task_id=None,
                ready_task_ids=[],
                blocked_task_ids=[],
                observations=[],
            ),
            context_packet=self._context_packet_builder.compress(
                session,
                task_board,
                latest_user_message=user_message,
            ),
            task_board=task_board,
            trace_id=trace.trace_id,
        )
        self._sync_loop_state(context)

        log_event(
            self._logger,
            "turn.started",
            trace_id=trace.trace_id,
            session_id=session.session_id,
            turn_number=turn_number,
            payload={"user_message": user_message},
        )
        await self._emit_hook(
            HookPoint.turn_start,
            session_id=session.session_id,
            trace_id=trace.trace_id,
            turn_number=turn_number,
            payload={"user_message": user_message},
        )

        while context.turn_result is None:
            if context.loop_state.step_number >= self._max_steps:
                fallback_action = FallbackAction(
                    reason="loop_exhausted",
                    user_message="I need to stop here because this turn reached the maximum number of steps.",
                )
                trace.fallback_reason = trace.fallback_reason or "runtime:loop_exhausted"
                context.turn_result = await self._finalize_turn(
                    context,
                    trace,
                    action=fallback_action,
                    error_summary="step limit reached before a terminal action was produced",
                )
                break

            decision = await self._decision_provider(context)
            trace.decisions.append(decision.model_copy(deep=True))
            log_event(
                self._logger,
                "decision.created",
                trace_id=trace.trace_id,
                session_id=session.session_id,
                turn_number=turn_number,
                payload={
                    "action_kind": decision.action.kind,
                    "next_task_label": decision.next_task_label,
                    "continue_loop": decision.continue_loop,
                },
            )
            await self._emit_hook(
                HookPoint.decision,
                session_id=session.session_id,
                trace_id=trace.trace_id,
                turn_number=turn_number,
                payload={
                    "action_kind": decision.action.kind,
                    "rationale": decision.rationale,
                    "next_task_label": decision.next_task_label,
                },
            )

            task = self._build_task(decision, context)
            task = task_board.add_task(task)
            task_record = TaskRecord(
                task_id=task.task_id,
                task_name=task.name,
                status=task.status.value,
                notes=["created"],
            )
            trace.task_records.append(task_record)
            self._sync_loop_state(context)
            await self._emit_hook(
                HookPoint.task,
                session_id=session.session_id,
                trace_id=trace.trace_id,
                turn_number=turn_number,
                payload={
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "status": task.status.value,
                    "depends_on": list(task.depends_on),
                },
            )

            capability = self._resolve_capability(decision.action)
            gate_result = self._hardening_gate.evaluate(
                decision.action,
                actor="main_agent",
                current_node=context.loop_state.current_node,
                topic=context.loop_state.current_node,
                user_message=user_message,
                observations=trace.observations,
                capability=capability,
                permission_policy=self._permission_policy,
                step_number=context.loop_state.step_number,
                max_steps=self._max_steps,
                trace=trace,
            )
            if gate_result.decision != "allow":
                task_board.mark_failed(task.task_id, gate_result.reason)
                task_record.status = TaskStatus.failed.value
                task_record.notes.append(f"gate_rejected:{gate_result.guardrail or gate_result.decision}")
                self._sync_loop_state(context)
                context.turn_result = await self._finalize_turn(
                    context,
                    trace,
                    action=gate_result.fallback_action
                    or FallbackAction(
                        reason=gate_result.guardrail or "gate_rejected",
                        user_message="I need to stop and re-check this request before I continue.",
                    ),
                    error_summary=gate_result.reason,
                )
                break

            current_task = task_board.next_ready()
            if current_task is None:
                context.turn_result = await self._finalize_turn(
                    context,
                    trace,
                    action=FallbackAction(
                        reason="no_ready_task",
                        user_message="I could not find a safe next step for this turn.",
                    ),
                    error_summary="task board had no ready task after decision creation",
                )
                break

            context.loop_state.step_number += 1
            task_record.status = current_task.status.value
            task_record.notes.append("started")
            self._sync_loop_state(context)

            action = decision.action
            if isinstance(action, CallToolAction):
                result = await self._execute_tool_task(
                    context,
                    trace,
                    task_record,
                    current_task,
                    action,
                )
                if result is not None:
                    context.turn_result = result
                    break
                continue

            if isinstance(action, CallSubAgentAction):
                result = await self._execute_sub_agent_task(
                    context,
                    trace,
                    task_record,
                    current_task,
                    action,
                )
                if result is not None:
                    context.turn_result = result
                    break
                continue

            task_board.mark_done(current_task.task_id)
            task_record.status = TaskStatus.done.value
            task_record.notes.append("completed")
            self._sync_loop_state(context)
            context.turn_result = await self._finalize_turn(
                context,
                trace,
                action=action,
                error_summary=None,
            )

        session.turn_count = turn_number
        if context.turn_result is not None:
            session.last_turn_status = context.turn_result.status

        log_event(
            self._logger,
            "turn.finished",
            trace_id=trace.trace_id,
            session_id=session.session_id,
            turn_number=turn_number,
            payload={
                "status": context.turn_result.status if context.turn_result is not None else "unknown",
                "completed_steps": (
                    context.turn_result.completed_steps if context.turn_result is not None else 0
                ),
            },
        )
        await self._emit_hook(
            HookPoint.turn_end,
            session_id=session.session_id,
            trace_id=trace.trace_id,
            turn_number=turn_number,
            payload={
                "status": context.turn_result.status if context.turn_result is not None else None,
                "completed_steps": context.turn_result.completed_steps if context.turn_result is not None else 0,
            },
        )
        return context.turn_result

    async def _execute_tool_task(
        self,
        context: TurnRuntimeContext,
        trace: TraceRecord,
        task_record: TaskRecord,
        task: TurnTask,
        action: CallToolAction,
    ) -> TurnResult | None:
        provider = self._registry.get(action.capability_name)
        if not isinstance(provider, ToolProvider):
            raise TypeError(f"Capability {action.capability_name!r} is not a tool provider.")

        invocation_id = f"inv-{trace.turn_number}-{len(trace.invocations) + 1}"
        await self._emit_hook(
            HookPoint.invocation,
            session_id=context.session.session_id,
            trace_id=trace.trace_id,
            turn_number=trace.turn_number,
            payload={
                "invocation_id": invocation_id,
                "task_id": task.task_id,
                "capability_name": provider.name,
                "status": "started",
            },
        )
        log_event(
            self._logger,
            "invocation.started",
            trace_id=trace.trace_id,
            session_id=context.session.session_id,
            turn_number=trace.turn_number,
            payload={
                "task_id": task.task_id,
                "capability_name": provider.name,
                "capability_kind": provider.descriptor.kind,
            },
        )

        try:
            observation = await provider.invoke(action.arguments)
        except Exception as exc:
            failure = InvocationRecord(
                invocation_id=invocation_id,
                task_id=task.task_id,
                capability_name=provider.name,
                capability_kind=provider.descriptor.kind,
                status="failed",
                arguments=dict(action.arguments),
                error=str(exc),
            )
            task.invocations.append(failure)
            trace.invocations.append(failure.model_copy(deep=True))
            task_record.invocation_ids.append(failure.invocation_id)
            context.task_board.mark_failed(task.task_id, str(exc))
            task_record.status = TaskStatus.failed.value
            task_record.notes.append("invocation_failed")
            log_event(
                self._logger,
                "invocation.failed",
                trace_id=trace.trace_id,
                session_id=context.session.session_id,
                turn_number=trace.turn_number,
                payload={
                    "task_id": task.task_id,
                    "capability_name": provider.name,
                    "error": str(exc),
                },
                level=logging.ERROR,
            )
            await self._emit_hook(
                HookPoint.invocation,
                session_id=context.session.session_id,
                trace_id=trace.trace_id,
                turn_number=trace.turn_number,
                payload={
                    "invocation_id": invocation_id,
                    "task_id": task.task_id,
                    "capability_name": provider.name,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            self._sync_loop_state(context)
            trace.fallback_reason = trace.fallback_reason or "runtime:invocation_failed"
            return await self._finalize_turn(
                context,
                trace,
                action=FallbackAction(
                    reason="invocation_failed",
                    user_message="I ran into a tool failure while working on that request.",
                ),
                error_summary=str(exc),
            )

        success = InvocationRecord(
            invocation_id=invocation_id,
            task_id=task.task_id,
            capability_name=provider.name,
            capability_kind=provider.descriptor.kind,
            status="succeeded",
            arguments=dict(action.arguments),
            observation_id=observation.observation_id,
        )
        task.invocations.append(success)
        trace.invocations.append(success.model_copy(deep=True))
        trace.observations.append(observation.model_copy(deep=True))
        context.loop_state.observations.append(observation.model_copy(deep=True))
        task_record.invocation_ids.append(success.invocation_id)
        context.task_board.mark_done(task.task_id)
        task_record.status = TaskStatus.done.value
        task_record.notes.append("completed")
        self._sync_loop_state(context)
        context.context_packet = self._context_packet_builder.compress(
            context.session,
            context.task_board,
            latest_user_message=context.context_packet.latest_user_message,
        )
        log_event(
            self._logger,
            "invocation.succeeded",
            trace_id=trace.trace_id,
            session_id=context.session.session_id,
            turn_number=trace.turn_number,
            payload={
                "task_id": task.task_id,
                "capability_name": provider.name,
                "observation_id": observation.observation_id,
            },
        )
        await self._emit_hook(
            HookPoint.invocation,
            session_id=context.session.session_id,
            trace_id=trace.trace_id,
            turn_number=trace.turn_number,
            payload={
                "invocation_id": invocation_id,
                "task_id": task.task_id,
                "capability_name": provider.name,
                "status": "succeeded",
                "observation_id": observation.observation_id,
            },
        )
        return None

    async def _execute_sub_agent_task(
        self,
        context: TurnRuntimeContext,
        trace: TraceRecord,
        task_record: TaskRecord,
        task: TurnTask,
        action: CallSubAgentAction,
    ) -> TurnResult | None:
        provider = self._registry.get(action.capability_name)
        if not isinstance(provider, SubAgentProvider):
            raise TypeError(f"Capability {action.capability_name!r} is not a sub-agent provider.")

        brief = self._build_brief(action, task, context)
        invocation_id = f"inv-{trace.turn_number}-{len(trace.invocations) + 1}"
        await self._emit_hook(
            HookPoint.invocation,
            session_id=context.session.session_id,
            trace_id=trace.trace_id,
            turn_number=trace.turn_number,
            payload={
                "invocation_id": invocation_id,
                "task_id": task.task_id,
                "capability_name": provider.name,
                "status": "started",
            },
        )
        log_event(
            self._logger,
            "invocation.started",
            trace_id=trace.trace_id,
            session_id=context.session.session_id,
            turn_number=trace.turn_number,
            payload={
                "task_id": task.task_id,
                "capability_name": provider.name,
                "capability_kind": provider.descriptor.kind,
            },
        )

        try:
            observation = await provider.invoke(brief)
        except Exception as exc:
            failure = InvocationRecord(
                invocation_id=invocation_id,
                task_id=task.task_id,
                capability_name=provider.name,
                capability_kind=provider.descriptor.kind,
                status="failed",
                arguments=brief.model_dump(mode="json"),
                error=str(exc),
            )
            task.invocations.append(failure)
            trace.invocations.append(failure.model_copy(deep=True))
            task_record.invocation_ids.append(failure.invocation_id)
            context.task_board.mark_failed(task.task_id, str(exc))
            task_record.status = TaskStatus.failed.value
            task_record.notes.append("invocation_failed")
            self._sync_loop_state(context)
            log_event(
                self._logger,
                "invocation.failed",
                trace_id=trace.trace_id,
                session_id=context.session.session_id,
                turn_number=trace.turn_number,
                payload={
                    "task_id": task.task_id,
                    "capability_name": provider.name,
                    "error": str(exc),
                },
                level=logging.ERROR,
            )
            trace.fallback_reason = trace.fallback_reason or "runtime:invocation_failed"
            return await self._finalize_turn(
                context,
                trace,
                action=FallbackAction(
                    reason="invocation_failed",
                    user_message="I ran into a specialist failure while working on that request.",
                ),
                error_summary=str(exc),
            )

        success = InvocationRecord(
            invocation_id=invocation_id,
            task_id=task.task_id,
            capability_name=provider.name,
            capability_kind=provider.descriptor.kind,
            status="succeeded",
            arguments=brief.model_dump(mode="json"),
            observation_id=observation.observation_id,
        )
        task.invocations.append(success)
        trace.invocations.append(success.model_copy(deep=True))
        trace.observations.append(observation.model_copy(deep=True))
        context.loop_state.observations.append(observation.model_copy(deep=True))
        task_record.invocation_ids.append(success.invocation_id)
        context.task_board.mark_done(task.task_id)
        task_record.status = TaskStatus.done.value
        task_record.notes.append("completed")
        self._sync_loop_state(context)
        context.context_packet = self._context_packet_builder.compress(
            context.session,
            context.task_board,
            latest_user_message=context.context_packet.latest_user_message,
        )
        log_event(
            self._logger,
            "invocation.succeeded",
            trace_id=trace.trace_id,
            session_id=context.session.session_id,
            turn_number=trace.turn_number,
            payload={
                "task_id": task.task_id,
                "capability_name": provider.name,
                "observation_id": observation.observation_id,
            },
        )
        await self._emit_hook(
            HookPoint.invocation,
            session_id=context.session.session_id,
            trace_id=trace.trace_id,
            turn_number=trace.turn_number,
            payload={
                "invocation_id": invocation_id,
                "task_id": task.task_id,
                "capability_name": provider.name,
                "status": "succeeded",
                "observation_id": observation.observation_id,
            },
        )
        return None

    def _build_task(self, decision: AgentDecision, context: TurnRuntimeContext) -> TurnTask:
        task_id = f"task-{context.session.session_id}-{len(context.task_board.tasks) + 1}"
        name = decision.next_task_label or self._default_task_name(decision)
        return TurnTask(
            task_id=task_id,
            name=name,
            description=decision.rationale,
        )

    def _build_brief(
        self,
        action: CallSubAgentAction,
        task: TurnTask,
        context: TurnRuntimeContext,
    ) -> SpecialistBrief:
        payload = dict(action.brief)
        payload.setdefault("brief_id", f"brief-{task.task_id}")
        payload.setdefault("task_id", task.task_id)
        payload.setdefault("goal", task.description or task.name)
        payload.setdefault("constraints", {})
        payload.setdefault("allowed_capabilities", [])
        payload.setdefault("context_packet", context.context_packet.model_dump(mode="json"))
        return SpecialistBrief.model_validate(payload)

    def _resolve_capability(self, action: object):
        if isinstance(action, (CallToolAction, CallSubAgentAction)):
            return self._registry.get(action.capability_name).descriptor
        return None

    async def _finalize_turn(
        self,
        context: TurnRuntimeContext,
        trace: TraceRecord,
        *,
        action,
        error_summary: str | None,
    ) -> TurnResult:
        if isinstance(action, ReplyToUserAction):
            status = "reply"
            message = action.message
        elif isinstance(action, AskClarificationAction):
            status = "clarification"
            message = action.question
        elif isinstance(action, FallbackAction):
            status = "fallback"
            message = action.user_message
            trace.fallback_reason = trace.fallback_reason or f"runtime:{action.reason}"
        else:
            raise TypeError(f"Unsupported terminal action: {type(action).__name__}")

        trace.terminal_state = status
        turn_result = TurnResult(
            session_id=context.session.session_id,
            turn_number=trace.turn_number,
            status=status,
            message=message,
            action=action,
            trace_id=trace.trace_id,
            completed_steps=context.loop_state.step_number,
            error_summary=error_summary,
        )
        self._trace_store.save(trace)
        if status == "fallback":
            log_event(
                self._logger,
                "turn.fallback",
                trace_id=trace.trace_id,
                session_id=context.session.session_id,
                turn_number=trace.turn_number,
                payload={
                    "reason": trace.fallback_reason,
                    "error_summary": error_summary,
                },
                level=logging.WARNING,
            )
            await self._emit_hook(
                HookPoint.fallback,
                session_id=context.session.session_id,
                trace_id=trace.trace_id,
                turn_number=trace.turn_number,
                payload={
                    "reason": trace.fallback_reason,
                    "error_summary": error_summary,
                },
            )
        return turn_result

    async def _emit_hook(
        self,
        point: HookPoint,
        *,
        session_id: str,
        trace_id: str,
        turn_number: int,
        payload: dict[str, object],
    ) -> None:
        if self._hook_bus is None:
            return
        await self._hook_bus.emit(
            point,
            HookEvent(
                hook_point=point,
                session_id=session_id,
                trace_id=trace_id,
                turn_number=turn_number,
                payload=payload,
            ),
        )

    @staticmethod
    def _memory_read_keys(session: SessionState) -> list[str]:
        return sorted(
            {f"session:{key}" for key in session.session_working_memory}
            | {f"durable:{key}" for key in session.durable_user_memory}
        )

    @staticmethod
    def _default_task_name(decision: AgentDecision) -> str:
        action = decision.action
        if isinstance(action, (CallToolAction, CallSubAgentAction)):
            return action.capability_name
        return action.kind

    @staticmethod
    def _sync_loop_state(context: TurnRuntimeContext) -> None:
        context.loop_state.current_task_id = context.task_board.current_task_id
        context.loop_state.ready_task_ids = list(context.task_board.ready_task_ids)
        context.loop_state.blocked_task_ids = list(context.task_board.blocked_task_ids)


__all__ = ["DecisionProvider", "SerialExecutor"]
