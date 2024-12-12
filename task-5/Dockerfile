FROM python:3.9-slim-buster

WORKDIR /app

RUN poetry install


COPY .. .

EXPOSE 8000

# Define the command to run your application
CMD ["python", "your_app.py"]

# TODO: Недоделал