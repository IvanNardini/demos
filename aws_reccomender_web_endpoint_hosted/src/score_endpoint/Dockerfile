
# Use an official Python runtime as parent image
FROM python:3
    
LABEL Scoring App = "Recommendation System - Python - Surprise"

# RUN apt-get update && apt-get install -y \
# python3-dev \
# build-essential    
        
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Upgrade pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 9999 available
EXPOSE 9999

#Entrypoint python exec
ENTRYPOINT [ "/bin/sh" ]

#Run flask application
CMD [ "run_server.sh" ]
