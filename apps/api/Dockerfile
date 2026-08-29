FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY apps/api/src ./apps/api/src
COPY packages/apex-core/src ./packages/apex-core/src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 apex
USER apex
EXPOSE 8000
CMD ["uvicorn", "apex_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
