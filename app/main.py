from fastapi import FastAPI

from app.api.routes import router
from app.core.config import DAMAGE_MODEL_PATH, DAMAGE_META_PATH, TYPE_MODEL_PATH, TYPE_META_PATH
from app.ml.model_loader import load_damage_bundle, load_type_bundle
from app.ml.predictor import Predictor


def create_app() -> FastAPI:
    app = FastAPI(title="Car Damage Detection Service", version="1.0.0")

    @app.on_event("startup")
    def _startup():
        damage = load_damage_bundle(DAMAGE_MODEL_PATH, DAMAGE_META_PATH)
        dtype = load_type_bundle(TYPE_MODEL_PATH, TYPE_META_PATH)
        app.state.predictor = Predictor(damage=damage, dtype=dtype)

    app.include_router(router)
    return app


app = create_app()