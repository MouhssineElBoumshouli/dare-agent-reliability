FROM volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9

# The upstream 2025-06-09 image contains pandas 2.2.0 but no Parquet engine.
# Pin the PyArrow release contemporary with the base image so every selected
# CSV, SQLite, and Parquet task runs in one recorded executor environment.
RUN /root/miniconda3/envs/sandbox-runtime/bin/python -m pip install \
      --no-cache-dir pyarrow==20.0.0 \
    && /root/miniconda3/envs/sandbox-runtime/bin/python -c \
      "import pyarrow; assert pyarrow.__version__ == '20.0.0'"
