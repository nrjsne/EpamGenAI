## Orders Management API (Module 7)

This module implements a small Orders Management REST API

### Features

- **POST `/orders`**: create a new order
- **GET `/orders/{id}`**: fetch single order
- **PUT `/orders/{id}`**: update an order
- **DELETE `/orders/{id}`**: delete an order
- **Database**: SQLite with SQLAlchemy models
- **Seeding**: script to seed 50 sample orders
- **Tests**: 12+ pytest tests that cover CRUD and edge cases

### Installation

```bash
cd module7
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the API

From the `module7` directory (with virtualenv activated):

```bash
cd /Users/nurzhaussyn/Documents/EpamGenAI/module7
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uvicorn orders_api.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Database Setup

The database is SQLite and is automatically created as `orders.db` in the `module7`
directory when the app starts.

To re-create tables manually in a REPL:

```python
from orders_api.database import init_db
init_db()
```

### Seeding 50 Sample Orders

To seed the database with 50 demo orders:

```bash
cd module7
python -m orders_api.seed
```

The script is idempotent for local usage: it clears existing orders and inserts 50 new
ones with varying statuses, amounts, and creation dates.

### API Endpoints

- **POST `/orders`**
  - Body: JSON matching `OrderCreate` (`customer_name`, `status`, `amount`)
  - Response: created order (`OrderRead`)

- **GET `/orders`**
  - Query parameters:
    - `status` (optional str)
  - Query parameters:
    - `page` (optional int, default 1)
    - `limit` (optional int, default 10, max 100)
    - `status` (optional str)
    - `amount_min` (optional float)
    - `amount_max` (optional float)
    - `date_from` (optional ISO datetime)
    - `date_to` (optional ISO datetime)
  - Response: `PaginatedOrders` structure:
    - `items`: list of orders
    - `total`: total matching orders
    - `page`: current page
    - `limit`: page size

  Example request:
  ```http
  GET /orders?page=2&limit=5&status=completed&amount_min=10&amount_max=100&date_from=2026-02-01T00:00:00&date_to=2026-02-13T23:59:59
  ```

  Example response:
  ```json
  {
    "items": [
      {
        "id": 12,
        "customer_name": "User 11",
        "status": "completed",
        "amount": 22.0,
        "created_at": "2026-02-02T10:00:00"
      }
      // ...more orders...
    ],
    "total": 17,
    "page": 2,
    "limit": 5
  }
  ```

- **GET `/orders/{id}`**
  - Response: `OrderRead` or 404

- **PUT `/orders/{id}`**
  - Body: partial update fields (`OrderUpdate`)
  - Response: updated order or 404

- **DELETE `/orders/{id}`**
  - Response: 204 No Content or 404

### Running Tests and Coverage

From the `module7` folder, with the virtualenv activated:

```bash
cd /Users/nurzhaussyn/Documents/EpamGenAI/module7
source .venv/bin/activate
pytest --cov=orders_api --cov-report=term-missing
```

Notes:

- Tests live under `orders_api/tests/` and are part of the `orders_api` package
  (via `__init__.py`), which allows imports like `from ..app import app`.
- If pytest cannot find the package, ensure you run it from `module7` where
  the `orders_api` package resides and that the virtualenv is active.

The test suite covers:

- Creating, reading, updating, and deleting orders
- Edge cases such as non-existent IDs, invalid payloads, and out-of-range pages

