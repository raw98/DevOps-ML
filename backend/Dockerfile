FROM python:3.8-slim-buster
EXPOSE 5000
COPY . /app
WORKDIR /app
RUN python -m pip install -r  requirements.txt
RUN pip install soundfile
CMD ["python","app.py"]
