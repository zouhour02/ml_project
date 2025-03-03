# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Supervisor to manage multiple processes
RUN apt-get update && apt-get install -y supervisor

# Create Supervisor config
RUN mkdir -p /var/log/supervisor

# Supervisor config file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for Flask and MLflow
EXPOSE 5000 5001

# Run Supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

