FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app/

# Expose port
EXPOSE 8000

# Run the application with shell expansion for PORT variable
CMD ["sh", "-c", "python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT"]