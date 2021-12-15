FROM python:3.8-slim-buster

EXPOSE 8080:8080
WORKDIR /app
RUN python -m pip install -r flask==1.1.2
VOLUME  . /app
CMD ["python", "main.py"]