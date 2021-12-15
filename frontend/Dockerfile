FROM python:3.8-slim-buster

EXPOSE 8080
WORKDIR /app
COPY . /app
RUN python -m pip install -r requirement.txt

CMD ["python", "app.py"]