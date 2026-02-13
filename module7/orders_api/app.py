from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from . import models, schemas
from .database import get_db, init_db


app = FastAPI(title="Orders API")


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
    db: Session = Depends(get_db),
):
    """
    List orders.
    """

    base_query = select(models.Order)

    result = db.execute(base_query)
    orders = result.scalars().all()

    return schemas.PaginatedOrders(
        items=orders,
    )

