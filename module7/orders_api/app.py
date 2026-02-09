from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from . import models, schemas
from .database import get_db, init_db


app = FastAPI(title="Orders API with Pagination and Filtering")


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database schema on startup."""
    init_db()


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/orders",
    response_model=schemas.OrderRead,
    status_code=status.HTTP_201_CREATED,
)
def create_order(order_in: schemas.OrderCreate, db: Session = Depends(get_db)):
    """
    Create a new order.

    This endpoint accepts an `OrderCreate` payload and persists it to the
    database. The `created_at` timestamp is generated on the server side.
    """
    db_order = models.Order(
        customer_name=order_in.customer_name,
        status=order_in.status,
        amount=order_in.amount,
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return db_order


@app.get("/orders/{order_id}", response_model=schemas.OrderRead)
def get_order(order_id: int, db: Session = Depends(get_db)):
    """Retrieve a single order by its ID."""
    db_order = db.get(models.Order, order_id)
    if not db_order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )
    return db_order


@app.put("/orders/{order_id}", response_model=schemas.OrderRead)
def update_order(
    order_id: int,
    order_in: schemas.OrderUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing order with new data."""
    db_order = db.get(models.Order, order_id)
    if not db_order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    update_data = order_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_order, field, value)
    db.commit()
    db.refresh(db_order)
    return db_order


@app.delete("/orders/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_order(order_id: int, db: Session = Depends(get_db)):
    """Delete an order by its ID."""
    db_order = db.get(models.Order, order_id)
    if not db_order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )
    db.delete(db_order)
    db.commit()
    return None


@app.get(
    "/orders",
    response_model=schemas.PaginatedOrders,
)
def list_orders(
    # page=1&limit=10 is a common pagination pattern:
    # - `page` is 1-based, easier for most API consumers
    # - `limit` controls page size; defaults to 10
    #
    # Suggestions for improvement:
    # - Enforce a sensible maximum limit (e.g. 100) to protect the API
    #   from unbounded queries.
    # - Consider returning navigation links (next/prev) in the payload
    #   for better API ergonomics.
    # - In high-traffic systems, consider cursor-based pagination for
    #   stable ordering and better performance on large datasets.
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by exact status value.",
        min_length=1,
        max_length=20,
    ),
    min_amount: Optional[float] = Query(
        None,
        ge=0,
        description="Filter by minimum order amount (inclusive).",
    ),
    max_amount: Optional[float] = Query(
        None,
        ge=0,
        description="Filter by maximum order amount (inclusive).",
    ),
    start_date: Optional[datetime] = Query(
        None,
        description="Filter by creation timestamp from this date/time (inclusive).",
    ),
    end_date: Optional[datetime] = Query(
        None,
        description="Filter by creation timestamp up to this date/time (inclusive).",
    ),
    db: Session = Depends(get_db),
):
    """
    List orders with pagination and optional filtering.

    Filtering options:
    - `status`: exact match on order status
    - `min_amount` / `max_amount`: inclusive range for the `amount` field
    - `start_date` / `end_date`: inclusive range for the `created_at` timestamp

    Security notes:
    - Filters are applied via SQLAlchemy's query builder, which uses
      parameter binding and protects against SQL injection.
    - Query parameters are validated and typed by FastAPI/Pydantic,
      which prevents malicious or malformed payloads from reaching
      the database layer.
    """
    conditions = []
    if status_filter:
        conditions.append(models.Order.status == status_filter)
    if min_amount is not None:
        conditions.append(models.Order.amount >= min_amount)
    if max_amount is not None:
        conditions.append(models.Order.amount <= max_amount)
    if start_date is not None:
        conditions.append(models.Order.created_at >= start_date)
    if end_date is not None:
        conditions.append(models.Order.created_at <= end_date)

    base_query = select(models.Order)
    count_query = select(func.count(models.Order.id))
    if conditions:
        base_query = base_query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

    total_items = db.scalar(count_query) or 0

    offset = (page - 1) * limit
    query = base_query.order_by(models.Order.created_at.desc()).offset(offset).limit(
        limit
    )
    result = db.execute(query)
    orders = result.scalars().all()

    total_pages = (total_items + limit - 1) // limit if total_items else 0

    return schemas.PaginatedOrders(
        page=page,
        limit=limit,
        total_items=total_items,
        total_pages=total_pages,
        items=orders,
    )

