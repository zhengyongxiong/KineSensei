# init_db.py
import os
from werkzeug.security import generate_password_hash
from flask import Flask
from models import db, User

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///dance_flask.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.drop_all()   # 每次初始化前清空旧表
    db.create_all()

    def add_user(username, password, is_admin):
        if not User.query.filter_by(username=username).first():
            db.session.add(User(
                username=username,
                password_hash=generate_password_hash(password),
                is_admin=is_admin
            ))

    add_user("admin", "1234", True)
    add_user("test", "1234", False)
    db.session.commit()
    print("✅ 数据库初始化完成。")
