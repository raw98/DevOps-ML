FROM python:3.8-slim-buster
EXPOSE 5000
COPY . /app
WORKDIR /app
RUN python -m pip install -r  requirement.txt
RUN pip install soundfile
VOLUME . .
CMD ["python","app.py"]
