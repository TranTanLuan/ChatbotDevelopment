# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# set PYTHON_VERSION var with value is 3.8.18
ARG PYTHON_VERSION=3.8.18
# down a base image (python:3.8.18) from Docker Hub, it contains python envs to run this app
FROM python:${PYTHON_VERSION} as base

# Prevents Python from writing pyc files.
# setup for env var
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
# setup for env var
ENV PYTHONUNBUFFERED=1

# set working dir in container is /app
WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
# install libs in requirements.txt. it use two mount types to optimize building process
# mount type cache: mount a cache dir into /root/.cache/pip to store downloaded packages, to avoid download again
# mount type bind: mount requirement.txt from server to container to avoid copy it to image.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the source code into the container.
# copy all files and dirs in current dir (where contains dockerfile) to working dir in container (/app)
COPY . .

# Expose the port that the application listens on.
# notify that app in container will listen at 8000 port. This port will need to be mapped to a port in server when container is runned.
EXPOSE 8000

# Run the application.
# specify command is executed when container is started.
CMD uvicorn main:app --reload --port 8000 --host 0.0.0.0
