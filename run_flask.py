"""Development Flask server with debug explicitly disabled."""
import os
from webapp import create_app

app = create_app()
if __name__ == "__main__":
    app.run(host=os.getenv("APP_HOST", "127.0.0.1"), port=int(os.getenv("APP_PORT", "8000")), debug=False)
