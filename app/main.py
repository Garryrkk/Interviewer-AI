from fastapi import FastAPI
from app import api

app = FastAPI(title="AI Meeting Assistant")

# include routers from features
app.include_router(api.router)

@app.get("/")
def root():
    return {"msg": "AI Meeting Assistant Backend Running 🚀"}
