"""Gunicorn settings, picked up automatically from the working directory.

Render routes to whatever port it puts in PORT, and gunicorn does not read that
variable: its default is 127.0.0.1:8000 whatever PORT says, so a service started
with a bare `gunicorn app:server` binds where nothing is listening and the deploy
fails its health check. Binding here rather than in the start command keeps that
command exactly as the assignment specifies it.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"
workers = 1          # Render sets WEB_CONCURRENCY=1 on a free instance, and two
                     # workers hold two copies of the dataset in 512 MB
threads = 4
timeout = 120          # the first request wakes a sleeping free instance
accesslog = "-"
