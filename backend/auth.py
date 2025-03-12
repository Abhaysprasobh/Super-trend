from datetime import timedelta, datetime 
from typing import Annotated 
from fastapi import APIRouter, Depends, HTTPException 
from pydantic import BaseModel 
from sqlalchemy.orm import Session 
from starlette import status 
from database import SessionLocal 
from models import Users 
from passlib.context import CryptContext 
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer 
from jose import jwt, JWTError

router = APIRouter(
	prefix='/auth',
	tags=['auth']
)
SECRET_KEY = '197b2c37c391bed93fe80344fe73b806947a65e36206e05a1a23c2fa12702fe3'
ALGORITHM = 'HS256'
bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

class CreateUserRequest(BaseModel):
	username: str
	password: str

class Token(BaseModel): 
	access_token: str
	token_type: str

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/" , status_code= status.HTTP_201_CREATED)
def create_user(db: db_dependency, create_user_request: CreateUserRequest):

	create_user_model = Users(
		username = create_user_request.username,
		hashed_password = bcrypt_context(create_user_request.password)
	)
	
	db.add(create_user_model)
	db.commit()