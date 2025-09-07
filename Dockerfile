# ---- Builder Stage ----
# In this stage, we install all dependencies, including build tools.
FROM python:3.9-slim as builder

WORKDIR /app

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install Python dependencies
# This creates a virtual environment which we will copy to the final image
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt


# ---- Final Stage ----
# This is the lean image that will run in production.
FROM python:3.9-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy your application code
COPY Backend/ ./Backend/
COPY Frontend/ ./Frontend/

# Set the PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Create and switch to a non-root user for security
RUN useradd -m -d /app appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Add a Python-based health check that is guaranteed to work
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; assert urllib.request.urlopen('http://localhost:5000/').status == 200"

# Set the command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "Backend.app:app"]
