import os

from flask import Flask
from .extensions import db
from .routes import main

def create_app(debug=False):
    app = Flask(__name__)
    from app.langchain_copy import LangChain

    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



    db.init_app(app)

    app.register_blueprint(main)
    langchain = LangChain()
    if debug:
        app.config['DEBUG'] = True


    app.config["LANGCHAIN"] = langchain
    return app