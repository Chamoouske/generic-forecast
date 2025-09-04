FROM python:3.12-slim as builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv pip compile -o requirements.txt pyproject.toml

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgomp1
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
