FROM python:3.9-slim

WORKDIR /app

# Install uv securely for blazing fast installations
RUN pip install --no-cache-dir uv

# Move requirements over and cache dependencies tightly using uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy Project
COPY . .

# Expose API port
EXPOSE 8000

# Exec into the backend logic
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
