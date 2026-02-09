"""
Utility script to seed the database with sample orders.

Run:
    uvicorn orders_api.app:app --reload
or execute this file directly:
    python -m orders_api.seed
"""

from datetime import datetime, timedelta
from random import choice, randint, uniform

from .database import SessionLocal, init_db
from .models import Order


STATUSES = ["pending", "processing", "shipped", "cancelled", "completed"]


def seed_orders(count: int = 50) -> None:
    """Seed the database with `count` sample orders."""
    init_db()
    db = SessionLocal()
    try:
        # Clear existing data to make seeding idempotent for local runs.
        db.query(Order).delete()

        now = datetime.utcnow()
        for i in range(count):
            created_at = now - timedelta(days=randint(0, 30))
            order = Order(
                customer_name=f"Customer {i + 1}",
                status=choice(STATUSES),
                amount=round(uniform(10.0, 500.0), 2),
                created_at=created_at,
            )
            db.add(order)
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    seed_orders()

