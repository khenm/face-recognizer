"""FastAPI server for face recognition and enrollment.

Endpoints:
    GET  /health          — Liveness probe
    GET  /people          — List enrolled people
    DELETE /people/{name} — Remove a person
    POST /recognize       — Recognize a face in an uploaded image
    POST /enroll          — Enroll a new person from 3-15 frames
"""

from .app import create_app

__all__ = ["create_app"]
