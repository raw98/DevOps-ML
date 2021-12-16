FROM python:3.8-slim-buster
EXPOSE 8081
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
WORKDIR /app
COPY . /app
CMD ["python", "app.py"]
