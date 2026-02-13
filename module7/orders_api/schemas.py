from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, NonNegativeFloat


class OrderCreate(BaseModel):
    """Payload schema for creating an order."""

    customer_name: str = Field(..., min_length=1, max_length=100)
    status: str = Field(..., min_length=1, max_length=20)
    amount: NonNegativeFloat


class OrderUpdate(BaseModel):
    """Payload schema for updating an order."""

    customer_name: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[str] = Field(None, min_length=1, max_length=20)
    amount: Optional[NonNegativeFloat] = None


class OrderRead(BaseModel):
    """Response schema for a single order."""

    id: int
    customer_name: str
    status: str
    amount: float
    created_at: datetime

    class Config:
        from_attributes = True


class PaginatedOrders(BaseModel):
    """
    Response schema for a paginated list of orders.
    """

    items: list[OrderRead]
    total: int
    page: int
    limit: int

