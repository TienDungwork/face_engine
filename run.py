import uvicorn
from app.core.config import app_config
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=app_config.API_HOST,
        port=app_config.API_PORT,
        workers=app_config.API_WORKERS,
        reload=app_config.API_RELOAD
    )
