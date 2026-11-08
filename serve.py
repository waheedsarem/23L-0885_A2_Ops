"""Production entry point using Waitress on Windows."""
import os
from waitress import serve
from webapp import create_app

app = create_app()
if __name__ == "__main__":
    serve(app, host=os.getenv("APP_HOST", "127.0.0.1"), port=int(os.getenv("APP_PORT", "8000")),
          threads=int(os.getenv("APP_THREADS", "4")))
