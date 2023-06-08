# Use the official Python base image with version 3.8
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY ./requirements.txt .
 
RUN pip install -r requirements.txt
 
COPY ./hrcapp .
 
EXPOSE 80
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]