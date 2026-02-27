import datetime
import logging
import sqlite3

# Fix Python 3.14 + IPython conflict: IPython's history module registers a
# "timestamp" SQLite converter that only handles ISO-format strings, but Feast
# stores Unix epoch integers. MLflow imports IPython at model-load time, which
# overwrites any previously registered converter. We solve this by:
# 1. Registering a converter that handles both formats
# 2. Patching register_converter so the "timestamp" slot can't be overwritten


def _timestamp_converter(val: bytes) -> datetime.datetime:
    text = val.decode() if isinstance(val, bytes) else val
    if text.isdigit():
        return datetime.datetime.fromtimestamp(int(text))
    return datetime.datetime.fromisoformat(text)


sqlite3.register_converter("timestamp", _timestamp_converter)

_original_register_converter = sqlite3.register_converter


def _guarded_register_converter(name: str, callable_: object) -> None:
    if name.lower() == "timestamp":
        return  # block IPython from overwriting our converter
    _original_register_converter(name, callable_)


sqlite3.register_converter = _guarded_register_converter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import features
from routes import fraud

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="Subbyx Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(features.router)
app.include_router(fraud.router)


@app.get("/")
def root():
    return {"message": "Subbyx Fraud Detection API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}
