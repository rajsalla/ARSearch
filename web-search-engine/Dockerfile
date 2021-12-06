FROM python:3.5-alpine

RUN apk add --no-cache gcc g++ make linux-headers libxml2 libxslt-dev libffi-dev openssl-dev jpeg-dev

ENV FLASK_APP index.py

WORKDIR ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apk --no-cache add curl

# RUN pip install -U scikit-learn # not working

COPY . .

CMD [ "flask", "run", "--host=0.0.0.0" ]

# build example : docker build -t web-search-engine .
# run example : docker run -p 5000:5000 -e "HOST=<ip>" -e "PORT=<port>" -e "USERNAME=<username>" -e "PASSWORD=<password>" web-search-engine
