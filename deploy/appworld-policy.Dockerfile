# Build from this directory: docker build -f appworld-policy.Dockerfile -t coevobench-appworld-policy:latest .
FROM python:3.11-slim
RUN pip install --no-cache-dir jinja2==3.1.6
RUN useradd --create-home --uid 1000 policy
USER policy
WORKDIR /tmp
CMD ["sleep", "infinity"]
