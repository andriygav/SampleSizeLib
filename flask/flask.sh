PYTHONPATH=. gunicorn --bind 0.0.0.0:$1 server:app
