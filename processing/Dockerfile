FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install transformers==4.9.1
RUN pip install elasticsearch==5.5.3
RUN pip install elasticsearch-dsl==5.4.0

# RUN apt-get install npm -y
# RUN npm install elasticdump==6.79.0 -g
# RUN apt-get update && apt-get upgrade -y && \
# RUN curl -sL https://deb.nodesource.com/setup_7.x | bash -
# RUN apt-get update -qq && apt-get install -y build-essential nodejs
# # RUN apt-get install -y nodejs 
# RUN apt-get install -y npm 
# RUN npm install elasticdump -g


ENTRYPOINT ["/bin/bash"]