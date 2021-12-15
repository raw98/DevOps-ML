FROM python:3.8-slim-buster
VOLUME . /app
EXPOSE 8080
WORKDIR /app
RUN python -m pip install -r flask==1.1.2

CMD ["python", "main.py"]