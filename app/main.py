from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.infrastructure.db.init_db import init_db
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield 

app = FastAPI(lifespan=lifespan)
app.include_router(router)