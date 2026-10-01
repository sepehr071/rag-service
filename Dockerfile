FROM python:3.13-slim
RUN groupadd -g 1313 appusr && useradd -m -d /home/appusr -u 1313 -g 1313 -s /bin/bash appusr
RUN pip install --no-cache-dir uv==0.11.16
WORKDIR /opt/
ENV PYTHONPATH=/opt
COPY --chown=appusr:appusr ./pyproject.toml ./uv.lock /opt/
RUN chown -R appusr:appusr /opt/
USER appusr
RUN uv sync --frozen --no-dev
COPY --chown=appusr:appusr ./app ./app/
# Worker count overridable at runtime via WEB_CONCURRENCY (default 4).
ENV WEB_CONCURRENCY=4
CMD ["sh", "-c", "uv run --no-dev gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY} -b 0.0.0.0:8000 --timeout 300"]
