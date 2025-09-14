# users/routes.py
from fastapi import APIRouter, HTTPException, Depends
from app.users.schemas import UserCreate, UserLogin, UserResponse, TokenResponse
from app.users.service import UserService

router = APIRouter()

@router.post("/register", response_model=UserResponse, tags=["Users"])
async def register_user(user: UserCreate):
    try:
        new_user = await UserService.create_user(user.username, user.email, user.password)
        return new_user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=TokenResponse, tags=["Users"])
async def login_user(credentials: UserLogin):
    user = await UserService.authenticate_user(credentials.email, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = UserService.create_access_token({"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse, tags=["Users"])
async def get_profile(current_user: UserResponse = Depends(UserService.authenticate_user)):
    return current_user
