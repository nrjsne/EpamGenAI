from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..app import app
from .. import models
from ..database import Base, get_db


SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test_orders.db"


engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def create_sample_orders(db: Session, count: int = 20) -> None:
    now = datetime.utcnow()
    for i in range(count):
        order = models.Order(
            customer_name=f"User {i}",
            status="completed" if i % 2 == 0 else "pending",
            amount=10.0 + i,
            created_at=now - timedelta(days=i),
        )
        db.add(order)
    db.commit()


def test_create_order(client: TestClient):
    payload = {
        "customer_name": "Alice",
        "status": "pending",
        "amount": 123.45,
    }
    response = client.post("/orders", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] > 0
    assert data["customer_name"] == "Alice"
    assert data["status"] == "pending"
    assert data["amount"] == 123.45


def test_get_order_by_id(client: TestClient):
    create = client.post(
        "/orders",
        json={"customer_name": "Bob", "status": "completed", "amount": 50.0},
    )
    order_id = create.json()["id"]

    response = client.get(f"/orders/{order_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == order_id
    assert data["customer_name"] == "Bob"


def test_get_order_not_found(client: TestClient):
    response = client.get("/orders/9999")
    assert response.status_code == 404


def test_update_order(client: TestClient):
    create = client.post(
        "/orders",
        json={"customer_name": "Carol", "status": "pending", "amount": 10.0},
    )
    order_id = create.json()["id"]

    response = client.put(
        f"/orders/{order_id}",
        json={"status": "completed", "amount": 20.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["amount"] == 20.0


def test_update_order_not_found(client: TestClient):
    response = client.put(
        "/orders/9999",
        json={"status": "completed"},
    )
    assert response.status_code == 404


def test_delete_order(client: TestClient):
    create = client.post(
        "/orders",
        json={"customer_name": "Dave", "status": "pending", "amount": 5.0},
    )
    order_id = create.json()["id"]

    delete_resp = client.delete(f"/orders/{order_id}")
    assert delete_resp.status_code == 204

    get_resp = client.get(f"/orders/{order_id}")
    assert get_resp.status_code == 404

def test_create_order_validation_error(client: TestClient):
    # amount cannot be negative
    payload = {
        "customer_name": "Invalid",
        "status": "pending",
        "amount": -1,
    }
    response = client.post("/orders", json=payload)
    assert response.status_code == 422


# --- Pagination & Filtering Tests ---
def test_list_orders_pagination_basic(client: TestClient):
    # Add 25 sample orders
    with TestingSessionLocal() as db:
        create_sample_orders(db, 25)
    resp = client.get("/orders?page=1&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 1
    assert data["limit"] == 10
    assert data["total"] == 25
    assert len(data["items"]) == 10

def test_list_orders_second_page(client: TestClient):
    with TestingSessionLocal() as db:
        create_sample_orders(db, 25)
    resp = client.get("/orders?page=2&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2
    assert data["limit"] == 10
    assert len(data["items"]) == 10

def test_list_orders_status_filter(client: TestClient):
    with TestingSessionLocal() as db:
        create_sample_orders(db, 20)
    resp = client.get("/orders?status=completed")
    assert resp.status_code == 200
    data = resp.json()
    assert all(order["status"] == "completed" for order in data["items"])

def test_list_orders_amount_range_filter(client: TestClient):
    with TestingSessionLocal() as db:
        create_sample_orders(db, 20)
    resp = client.get("/orders?amount_min=15&amount_max=25")
    assert resp.status_code == 200
    data = resp.json()
    for order in data["items"]:
        assert 15 <= order["amount"] <= 25

def test_list_orders_date_range_filter(client: TestClient):
    with TestingSessionLocal() as db:
        create_sample_orders(db, 10)
    date_from = (datetime.utcnow() - timedelta(days=5)).isoformat()
    date_to = datetime.utcnow().isoformat()
    resp = client.get(f"/orders?date_from={date_from}&date_to={date_to}")
    assert resp.status_code == 200
    data = resp.json()
    for order in data["items"]:
        assert order["created_at"] >= date_from
        assert order["created_at"] <= date_to

def test_pagination_edge_page_beyond_total(client: TestClient):
    with TestingSessionLocal() as db:
        create_sample_orders(db, 10)
    resp = client.get("/orders?page=5&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 5
    assert len(data["items"]) == 0

def test_invalid_pagination_params(client: TestClient):
    resp = client.get("/orders?page=0&limit=200")
    assert resp.status_code == 422

