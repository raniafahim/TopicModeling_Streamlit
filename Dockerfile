# syntax = docker/dockerfile:1.2
FROM python:3.10

WORKDIR /app
COPY requirements.txt /tmp/

RUN --mount=type=cache,target=/var/cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/var/cache/pip pip install --requirement /tmp/requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit.py","--server.port", "3838"]