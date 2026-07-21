from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import router as api_router
from app.core.config import app_config
from app.core.model_manager import insightface_model_manager
from app.core.local_db import local_db_manager
from app.core.task_manager import task_manager
from app.core.smart_face_pool import init_smart_face_pool, close_smart_face_pool
import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the models
    await insightface_model_manager.load_model()
    await init_smart_face_pool()

    # Initial sync based on RELOAD_DB config
    if app_config.RELOAD_DB:
        try:
            print("Starting initial sync...")
            await local_db_manager.sync_persons()
            print("Initial sync completed successfully")
        except Exception as e:
            print(f"Initial sync failed: {str(e)}")
            print("Continuing without initial sync")
    else:
        print("Skipping initial sync (RELOAD_DB=false)")

    # Start scheduled tasks
    await task_manager.start_scheduled_tasks()

    yield

    # Shutdown: stop tasks and clean up the models
    await task_manager.stop_all_tasks()
    await close_smart_face_pool()
    await insightface_model_manager.close_model()

app = FastAPI(
    title=app_config.APP_NAME,
    version=app_config.APP_VERSION,
    lifespan=lifespan
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=app_config.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=62070)
