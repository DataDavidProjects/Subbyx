import logging
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
