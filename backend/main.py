from fastapi import FastAPI
from .database import engine, Base
from .routes.auth import router as auth_router
from .routes.stock import router as stock_router

app = FastAPI()

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Tables already exist: {e}")

# Include routers
app.include_router(auth_router)
app.include_router(stock_router)

@app.get("/")
def read_root():
    return {"message": "Stock API Service"}