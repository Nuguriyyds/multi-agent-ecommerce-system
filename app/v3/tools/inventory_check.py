from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import uuid4

from app.v3.models import (
    CapabilityDescriptor,
    CapabilityKind,
    InventoryAvailability,
    InventoryCheckRequest,
    InventoryStatus,
    Observation,
    Product,
)
from app.v3.registry import ToolProvider

from .seed_data import find_product, get_seed_catalog

_INVENTORY_CHECK_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string"},
    },
    "required": ["sku"],
    "additionalProperties": False,
}

_INVENTORY_CHECK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string"},
        "product_name": {"type": "string"},
        "status": {"type": "string"},
        "quantity": {"type": "integer"},
        "is_available": {"type": "boolean"},
    },
    "required": ["sku", "product_name", "status", "quantity", "is_available"],
}


def inventory_check(
    sku: str,
    *,
    catalog: Sequence[Product] | None = None,
) -> InventoryStatus:
    request = InventoryCheckRequest(sku=sku)
    product = find_product(request.sku, catalog)

    if product.stock <= 0:
        availability = InventoryAvailability.out_of_stock
    elif product.stock <= product.low_stock_threshold:
        availability = InventoryAvailability.low_stock
    else:
        availability = InventoryAvailability.in_stock

    return InventoryStatus(
        sku=product.sku,
        product_name=product.name,
        status=availability,
        quantity=product.stock,
        is_available=availability is not InventoryAvailability.out_of_stock,
        low_stock_threshold=product.low_stock_threshold,
    )


class InventoryCheckProvider(ToolProvider):
    def __init__(self, *, catalog: Sequence[Product] | None = None) -> None:
        self._catalog = [product.model_copy(deep=True) for product in (catalog or get_seed_catalog())]
        super().__init__(
            CapabilityDescriptor(
                name="inventory_check",
                kind=CapabilityKind.tool,
                input_schema=_INVENTORY_CHECK_INPUT_SCHEMA,
                output_schema=_INVENTORY_CHECK_OUTPUT_SCHEMA,
                timeout=2.0,
                permission_tag="inventory.read",
                description="Check local mock inventory status for a SKU.",
            )
        )
        self._logger = logging.getLogger(__name__)

    async def invoke(self, args: dict[str, object]) -> Observation:
        self._logger.info("inventory_check start args=%s", args)
        try:
            request = InventoryCheckRequest.model_validate(args)
            status = inventory_check(request.sku, catalog=self._catalog)
        except Exception:
            self._logger.exception("inventory_check failed")
            raise

        observation = Observation(
            observation_id=f"obs-{uuid4().hex[:12]}",
            source=self.name,
            summary=f"{status.product_name} inventory is {status.status.value}.",
            payload=status.model_dump(mode="json"),
            evidence_source=f"tool:{self.name}",
        )
        self._logger.info(
            "inventory_check success sku=%s status=%s observation_id=%s",
            status.sku,
            status.status.value,
            observation.observation_id,
        )
        return observation


__all__ = ["InventoryCheckProvider", "inventory_check"]
