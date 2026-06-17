
# Stage - 1 Builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage - 2 Runtime
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY streamlit_app.py .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "sales_intelligence.main:app", "--host", "0.0.0.0", "--port", "8000"]