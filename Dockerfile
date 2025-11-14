FROM rayproject/ray:2.51.1-py310-cu128

# Set working directory
WORKDIR /app

# Copy training script
COPY train.py /app/train.py

# Install additional dependencies
RUN pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.24.0 \
    boto3 \
    s3fs

# The training script will be executed by the RayJob
CMD ["python", "/app/train.py"]
