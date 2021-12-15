FROM python:3.8-slim-buster
VOLUME . /app
EXPOSE 5000
WORKDIR /app
COPY . /app
RUN python -m pip install -r flask==1.1.2 librosa==0.6.0 numba==0.48 numpy keras tensorflow scikit-learn pillow matplotlib
RUN pip install soundfile
CMD ["python","app.py"]
