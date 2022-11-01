# base image: UCSD ETS scipy-ml-notebook
FROM ucsdets/scipy-ml-notebook AS base

# Update apt-get
RUN apt-get update && apt-get install git

# Clone latest executable from repository
RUN git clone https://github.com/Barry0121/dsc180-graph-neural-net.git