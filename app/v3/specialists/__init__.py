"""V3 fixed specialist runtime primitives and role registry."""

from .base import (
    Specialist,
    SpecialistCapabilityTypeError,
    SpecialistPermissionError,
    SpecialistRoleMismatch,
)
from .candidate_analysis import (
    CandidateAnalysisItem,
    CandidateAnalysisPayload,
    CandidateAnalysisSpecialist,
    CandidateFitReason,
    register_candidate_analysis_prompt,
)
from .comparison import (
    ALLOWED_COMPARISON_DIMENSIONS,
    ComparisonObservationRef,
    ComparisonSpecialist,
    DomainComparisonPayload,
    register_comparison_prompt,
)
from .recommendation_rationale import (
    RationaleItem,
    RecommendationRationalePayload,
    RecommendationRationaleSpecialist,
    register_recommendation_rationale_prompt,
)
from .shopping_brief import (
    BudgetSlot,
    ShoppingBriefPayload,
    ShoppingBriefSpecialist,
    register_shopping_brief_prompt,
)
from .team import AgentTeam, SpecialistAlreadyRegistered, SpecialistNotFound

__all__ = [
    "ALLOWED_COMPARISON_DIMENSIONS",
    "AgentTeam",
    "BudgetSlot",
    "CandidateAnalysisItem",
    "CandidateAnalysisPayload",
    "CandidateAnalysisSpecialist",
    "CandidateFitReason",
    "ComparisonObservationRef",
    "ComparisonSpecialist",
    "DomainComparisonPayload",
    "RationaleItem",
    "RecommendationRationalePayload",
    "RecommendationRationaleSpecialist",
    "ShoppingBriefPayload",
    "ShoppingBriefSpecialist",
    "Specialist",
    "SpecialistAlreadyRegistered",
    "SpecialistCapabilityTypeError",
    "SpecialistNotFound",
    "SpecialistPermissionError",
    "SpecialistRoleMismatch",
    "register_candidate_analysis_prompt",
    "register_comparison_prompt",
    "register_recommendation_rationale_prompt",
    "register_shopping_brief_prompt",
]
