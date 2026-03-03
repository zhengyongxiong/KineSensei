from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)  # ✅ 匹配 app.py 中 Course 创建逻辑
    video_url = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    levels = db.relationship("Level", backref="course", cascade="all, delete-orphan")

class Level(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("course.id"), nullable=False)
    
    level_number = db.Column(db.Integer, nullable=False)  # ✅ 替代 level_index，支持修改
    name = db.Column(db.String(128), nullable=False)      # ✅ 新增：关卡显示名称（自定义）

    action_name = db.Column(db.String(128), nullable=False)
    video_url = db.Column(db.String(256))
    require_upper = db.Column(db.String(64))
    require_lower = db.Column(db.String(64))
    pass_condition = db.Column(db.String(16), default="both")  # both / upper / lower
    pass_score = db.Column(db.Integer, default=80)


class Progress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    level_id = db.Column(db.Integer, db.ForeignKey("level.id"), nullable=False)
    passed = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# 仅展示新增表
class Video(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    filename  = db.Column(db.String(256), nullable=False)
    uploader  = db.Column(db.Integer, db.ForeignKey('user.id'))
    part      = db.Column(db.String(10))       # 'upper' or 'lower'
    label     = db.Column(db.String(64))       # 动作名称
    created   = db.Column(db.DateTime, default=datetime.utcnow)

class TrainingJob(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    part      = db.Column(db.String(10))              # 'upper' / 'lower'
    status    = db.Column(db.String(20), default='PENDING')  # PENDING/RUNNING/DONE/FAIL
    progress  = db.Column(db.Integer, default=0)      # 0-100
    metrics   = db.Column(db.Text)                    # JSON：val_acc/cls_report 等
    model_path= db.Column(db.String(256))
    log_path  = db.Column(db.String(256))
    started   = db.Column(db.DateTime)
    finished  = db.Column(db.DateTime)
