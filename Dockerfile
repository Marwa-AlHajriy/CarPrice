FROM python:3.11  
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

ARG MODEL_URL

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
 && echo "Downloading model files..." \
 && curl -L -o used_car_price_model.pkl "$MODEL_URL"

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

