FROM python:3.11-slim as builder
RUN pip install uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv pip compile -o requirements.txt pyproject.toml

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
