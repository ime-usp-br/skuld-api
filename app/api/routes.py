# app/api/routes.py
from fastapi import FastAPI

app = FastAPI(
    title="Skuld API",
    description="Microserviço preditivo para alocação de salas do IME-USP",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Skuld API está viva e respirando!"}