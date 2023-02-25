FROM python:latest

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY detect_iot.py ./
COPY UNSW_NB15_training-set.csv ./

CMD ["python", "./detect_iot.py"]

EXPOSE 8080
