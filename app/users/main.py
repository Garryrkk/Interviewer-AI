import logging
from fastapi import FastAPI

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

try:
    # Import user routes
    from app.users import routes as user_routes
    
    # Include the router
    app.include_router(
        user_routes.router,
        prefix="/api/v1/users",
        tags=["Users"]
    )
    logger.info("User routes successfully loaded")
    
except ImportError as e:
    logger.error(f"Failed to import user routes: {e}")
    logger.error("Make sure the following exists:")
    logger.error("1. app/users/routes.py file")
    logger.error("2. app/users/__init__.py file") 
    logger.error("3. app/__init__.py file")
    logger.error("4. 'router' variable is defined in app/users/routes.py")
except AttributeError as e:
    logger.error(f"Router attribute not found: {e}")
    logger.error("Make sure 'router' is defined in app/users/routes.py")
except Exception as e:
    logger.error(f"Unexpected error loading user routes: {e}")

# Alternative approach if the above doesn't work
# Uncomment the following if you need to import from a different structure:

# try:
#     import sys
#     import os
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#     from users import routes as user_routes
#     app.include_router(user_routes.router, prefix="/api/v1/users", tags=["Users"])
# except ImportError as e:
#     logger.error(f"Alternative import failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)