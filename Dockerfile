# Base Image
FROM python:3.13.0rc2-slim

# Install system dependencies 
RUN apt-get update && \
    apt-get install --no-install-recommends -y g++=4:10.2.1* ffmpeg=7:4.3.5* libsm6=2:1.2.3* libxext6=2:1.3.3* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt /.
RUN pip3 install --no-cache-dir wheel -r requirements.txt

# Add application code to the container’s file system
RUN mkdir -p /home/app
WORKDIR /home/app
COPY super_mario_bros .

# Specify execution environment & run application
ENTRYPOINT ["python3", "train.py"]
