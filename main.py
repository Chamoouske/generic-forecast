# main.py
# Ponto de entrada da aplicação FastAPI.

from fastapi import FastAPI
from src.infrastructure.api.routes import router

app = FastAPI(
    title="Generic Time Series Forecast API (Clean Architecture)",
    description="API para treino e previsão de séries temporais, seguindo Clean Architecture.",
    version="0.2.0",
)

app.include_router(router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Generic Time Series Forecast API!"}

if __name__ == "__main__":
    import uvicorn
    print("Para rodar a API localmente, execute no terminal:")
    print("uvicorn main:app --reload")
    print("Acesse http://127.0.0.1:8000/docs para a documentação interativa.")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
