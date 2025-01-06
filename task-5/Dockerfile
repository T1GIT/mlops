FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-root

COPY ./task-5/app.py ./task-5/app.py
COPY ./task-5/transformers.py ./task-5/transformers.py

EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "task-5/app.py", "--server.port=8501"]