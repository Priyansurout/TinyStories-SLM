# Use a standard Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all your application files (like inference_server.py) into the container
COPY . .

# Tell the container to expose port 7860 (the default for HF Spaces)
EXPOSE 7860

# The command to run your Flask application
# It uses the app_file name from your README.md
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:7860", "inference_server:app"]
