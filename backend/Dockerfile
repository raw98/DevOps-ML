FROM python:3.8-slim-buster

EXPOSE 5000
WORKDIR /app
COPY . /app
RUN python -m pip install -r requirement.txt 
RUN pip install soundfile
CMD ["python","app.py"]
