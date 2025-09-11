from fastapi import APIRouter
from app.quick_respond import routes as qr_routes
from app.summarization import routes as sum_routes
from app.insights import routes as insights_routes
from app.voice_recognition import routes as vr_routes
from app.image_recognition import routes as ir_routes
from app.agenda import routes as agenda_routes
from app.invisibility import routes as inv_routes

router = APIRouter()

# Mount feature routers
router.include_router(qr_routes.router, prefix="/quick-respond", tags=["Quick Respond"])
router.include_router(sum_routes.router, prefix="/summarization", tags=["Summarization"])
router.include_router(insights_routes.router, prefix="/insights", tags=["Insights"])
router.include_router(vr_routes.router, prefix="/voice", tags=["Voice Recognition"])
router.include_router(ir_routes.router, prefix="/image", tags=["Image Recognition"])
router.include_router(agenda_routes.router, prefix="/agenda", tags=["Agenda"])
router.include_router(inv_routes.router, prefix="/invisibility", tags=["Invisibility"])
