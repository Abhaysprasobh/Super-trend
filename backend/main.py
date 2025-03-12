from fastapi import FastAPI, HTTPException, Path, APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError

from sqlalchemy import  create_engine,Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from  sqlalchemy.orm import sessionmaker


import auth 
app = FastAPI()
app.include_router(auth.router)
DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

#  Models

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    number = Column(Integer, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    Name = Column(String, index=True)
    
    


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



