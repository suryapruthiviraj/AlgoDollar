from fastapi import APIRouter

from app.api.routes.allocation import router as allocation_router
from app.api.routes.audit import router as audit_router
from app.api.routes.health import router as health_router
from app.api.routes.markets import router as markets_router
from app.api.routes.portfolio import router as portfolio_router
from app.api.routes.research import router as research_router
from app.api.routes.risk import router as risk_router
from app.api.routes.settings import router as settings_router
from app.api.routes.strategies import router as strategies_router
from app.api.routes.trades import router as trades_router

api_router = APIRouter()

api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(allocation_router, prefix="/allocation", tags=["allocation"])
api_router.include_router(trades_router, prefix="/trades", tags=["trades"])
api_router.include_router(strategies_router, prefix="/strategies", tags=["strategies"])
api_router.include_router(markets_router, prefix="/markets", tags=["markets"])
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])
api_router.include_router(research_router, prefix="/research", tags=["research"])
# Execution decisions, including every refusal and the gate that caused it.
# Without this the UI can only report an absence, which is indistinguishable
# from a quiet market, a dead feed and an engaged kill switch.
api_router.include_router(audit_router, prefix="/audit", tags=["audit"])
api_router.include_router(risk_router, prefix="/risk", tags=["risk"])
