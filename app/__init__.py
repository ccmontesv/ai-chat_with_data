import os
from flask import Flask

def create_app():
    template_path = os.path.join(os.getcwd(), "templates")
    app = Flask(__name__, template_folder=template_path)
    from .routes import main
    app.register_blueprint(main)
    return app
