
FROM python:3.10-slim-bullseye

RUN mkdir -p prediction-module

WORKDIR prediction-module

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]
