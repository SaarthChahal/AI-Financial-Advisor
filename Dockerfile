# Use a minimal Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app and requirements
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the ADK web UI on port 8080
EXPOSE 8080

# Run the ADK web app
CMD ["adk", "web", "app", "--port", "8080", "--host", "0.0.0.0"]
