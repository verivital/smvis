"""WSGI entrypoint for production servers (gunicorn / Cloud Run).

Usage:
    gunicorn smvis.wsgi:server --bind 0.0.0.0:8080

``create_app()`` builds the Dash app and, as a side effect, registers its routes
and callbacks on the underlying Flask server. We expose that Flask object as
``server`` for the WSGI host. (Local development still uses ``python -m smvis``,
which calls ``app.run`` itself.)
"""
from smvis.app import create_app

app = create_app()
server = app.server
