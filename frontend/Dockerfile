FROM python:3.8-slim-buster

EXPOSE 8080:8080
COPY . /app
WORKDIR /app
RUN python -m pip install -r requirement.txt
VOLUME . .
CMD ["python", "main.py"]