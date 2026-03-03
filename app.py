from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps
import hashlib
import os
import json
from recognizer_api import classify_pose_from_image as classify_pose
from recognizer_api import pose_detector, classify_confidence_from_keypoints
import uuid

from models import db, User, Course, Level, Progress
import base64
import cv2
import numpy as np
from PIL import Image
import io
from flask import Response

app = Flask(__name__)
app.secret_key = "devkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///dance_flask.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
UPLOAD_FOLDER = "static/learn"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db.init_app(app)  # ✅ 正确绑定 app

with app.app_context():
    db.create_all()  # ✅ 自动创建数据库表

# ------------------- 权限装饰器 -------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            flash("需要管理员权限")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrapper

@app.route("/setup_db")
def setup_db():
    # 1. 设置管理员
    admin_user = User.query.filter_by(username="test").first()
    if not admin_user and "user_id" in session:
        admin_user = User.query.get(session["user_id"])
    
    msg = []
    if admin_user:
        admin_user.is_admin = True
        db.session.commit()
        msg.append(f"用户 {admin_user.username} 已设置为管理员")
    else:
        msg.append("未找到用户 test，请先注册或登录")

    # 2. 初始化课程数据
    if Course.query.count() == 0:
        c1 = Course(title="Popping 基础入门", description="适合零基础的 Popping 课程", video_url="")
        c2 = Course(title="Locking 进阶", description="Locking 核心动作教学", video_url="")
        db.session.add_all([c1, c2])
        db.session.commit()
        
        # 添加关卡
        l1 = Level(
            course_id=c1.id,
            level_number=1,
            name="Fresno 基础",
            action_name="fresno",
            video_url="",
            require_upper="fresno",
            pass_condition="upper",
            pass_score=80
        )
        db.session.add(l1)
        db.session.commit()
        msg.append("已初始化示例课程数据")
    else:
        msg.append("课程数据已存在，跳过初始化")

    return "<br>".join(msg) + "<br><a href='/'>返回首页</a>"

# ------------------- 路由：基础功能 -------------------
@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        if User.query.filter_by(username=u).first():
            flash("用户名已存在")
            return redirect(url_for("register"))
        user = User(username=u)
        user.set_password(p)
        db.session.add(user)
        db.session.commit()
        flash("注册成功，请登录")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        user = User.query.filter_by(username=u).first()
        if not user or not user.check_password(p):
            flash("登录失败，请检查用户名或密码")
            return redirect(url_for("login"))
        session["user_id"] = user.id
        session["username"] = user.username
        session["is_admin"] = user.is_admin
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/admin/users")
@admin_required
def list_users():
    users = User.query.all()
    return render_template("users.html", users=users)

@app.route("/admin/users/<int:user_id>/promote", methods=["POST"])
@admin_required
def promote_user(user_id):
    user = User.query.get_or_404(user_id)
    user.is_admin = True
    db.session.commit()
    return redirect(url_for("list_users"))

@app.route("/admin/users/<int:user_id>/reset", methods=["POST"])
@admin_required
def reset_password(user_id):
    new_pw = request.form["new_password"]
    user = User.query.get_or_404(user_id)
    user.set_password(new_pw)
    db.session.commit()
    flash("密码重置成功")
    return redirect(url_for("list_users"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for("list_users"))


# ------------------- 视频上传 -------------------
@app.route("/admin/videos", methods=["GET", "POST"])
@admin_required
def manage_videos():
    if request.method == "POST":
        file = request.files["video"]
        if file:
            content = file.read()
            ext = os.path.splitext(file.filename)[1]
            hashname = hashlib.sha256(content).hexdigest()[:10] + ext
            save_path = os.path.join(UPLOAD_FOLDER, hashname)
            with open(save_path, "wb") as f:
                f.write(content)
            flash(f"已上传：{hashname}")
        return redirect(url_for("manage_videos"))
    
    files = os.listdir(UPLOAD_FOLDER)
    files.sort()
    return render_template("video_manager.html", files=files)

@app.route("/admin/videos/delete/<filename>", methods=["POST"])
@admin_required
def delete_video(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        flash(f"已删除：{filename}")
    return redirect(url_for("manage_videos"))

# ------------------- 课程列表 -------------------
@app.route("/courses")
@admin_required
def list_courses():
    courses = Course.query.order_by(Course.created_at.desc()).all()
    return render_template("courses.html", courses=courses)

@app.route("/courses/add", methods=["POST"])
@admin_required
def add_course():
    title = request.form["title"]
    description = request.form.get("description", "")
    video_url = request.form.get("video_url", "")
    db.session.add(Course(title=title, description=description, video_url=video_url))
    db.session.commit()
    return redirect(url_for("list_courses"))

@app.route("/courses/<int:course_id>")
@admin_required
def course_detail(course_id):
    course = Course.query.get_or_404(course_id)
    levels = Level.query.filter_by(course_id=course_id).order_by(Level.level_number).all()

    # 自动建议编号
    last_level = Level.query.filter_by(course_id=course_id).order_by(Level.level_number.desc()).first()
    next_number = (last_level.level_number + 1) if last_level else 1

    try:
        with open("mlp/label_map_upper.json", "r", encoding="utf-8") as f:
            upper_map = json.load(f)
    except Exception as e:
        upper_map = {}
        print(f"读取 upper_map 错误: {e}")

    try:
        with open("mlp/label_map_lower.json", "r", encoding="utf-8") as f:
            lower_map = json.load(f)
    except Exception as e:
        lower_map = {}
        print(f"读取 lower_map 错误: {e}")

    return render_template(
        "course_levels.html",
        course=course,
        levels=levels,
        upper_map=upper_map,
        lower_map=lower_map,
        next_number=next_number
    )


@app.route("/courses/<int:course_id>/levels/add", methods=["POST"])
@admin_required
def add_level(course_id):
    data = request.form

    try:
        name = data.get("name", "").strip()
        level_number = int(data.get("level_number", 0))
        action_types = data.getlist("action_type[]")
        action_labels = data.getlist("action_label[]")
        video_url = data.get("video_url", "").strip()
        pass_score = int(data.get("pass_score", 80))
    except Exception as e:
        flash(f"提交数据错误: {e}")
        return redirect(url_for("course_detail", course_id=course_id))

    if not name:
        flash("关卡名称不能为空")
        return redirect(url_for("course_detail", course_id=course_id))

    if level_number <= 0:
        flash("关卡编号必须大于 0")
        return redirect(url_for("course_detail", course_id=course_id))

    if Level.query.filter_by(course_id=course_id, level_number=level_number).first():
        flash(f"该课程中编号 {level_number} 已存在，请更换编号")
        return redirect(url_for("course_detail", course_id=course_id))

    if len(action_types) != len(action_labels):
        flash("动作类型与标签数量不一致")
        return redirect(url_for("course_detail", course_id=course_id))

    sequence_parts = []
    require_upper, require_lower = "", ""
    for t, label in zip(action_types, action_labels):
        if t == "upper":
            require_upper = label
        elif t == "lower":
            require_lower = label
        sequence_parts.append(f"{t}:{label}")
    action_name = " | ".join(sequence_parts)

    pass_condition = (
        "both" if require_upper and require_lower else
        "upper" if require_upper else
        "lower" if require_lower else "both"
    )

    # ✅ 新增：处理视频文件上传
    video_file = request.files.get("video")
    video_filename = ""
    if video_file and video_file.filename:
        filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
        save_path = os.path.join(app.root_path, 'static/learn', filename)
        video_file.save(save_path)
        video_filename = filename

    # ✅ 创建关卡对象，加入 video_filename
    level = Level(
        course_id=course_id,
        level_number=level_number,
        name=name,
        action_name=action_name,
        video_url=video_url,
        pass_score=pass_score,
        require_upper=require_upper,
        require_lower=require_lower,
        pass_condition=pass_condition,
        video_filename=video_filename
    )

    db.session.add(level)
    db.session.commit()

    return redirect(url_for("course_detail", course_id=course_id))




@app.route("/courses/<int:course_id>/levels/<int:level_id>/delete", methods=["POST"])
@admin_required
def delete_level(course_id, level_id):
    level = Level.query.get_or_404(level_id)
    db.session.delete(level)
    db.session.commit()
    return redirect(url_for("course_detail", course_id=course_id))


# ------------------- 学习入口 -------------------
@app.route("/learn")
@login_required
def learn_courses():
    courses = Course.query.order_by(Course.created_at.desc()).all()
    return render_template("learn_courses.html", courses=courses)

@app.route("/learn/<int:course_id>")
@login_required
def learn_course(course_id):
    course = Course.query.get_or_404(course_id)
    levels = Level.query.filter_by(course_id=course_id).order_by(Level.level_number).all()

    progresses = Progress.query.filter_by(user_id=session["user_id"]).all()
    passed_set = {p.level_id for p in progresses if p.passed}

    unlocked = set()
    for l in levels:
        if l.level_number == 1:
            unlocked.add(l)
        else:
            prev = next((x for x in levels if x.level_number == l.level_number - 1), None)
            if prev and prev.id in passed_set:
                unlocked.add(l)

    progress_dict = {p.level_id: p for p in progresses}
    levels_display = [l for l in levels if l in unlocked]

    return render_template("learn_levels.html", course=course, levels=levels_display, progress=progress_dict)

@app.route("/courses/<int:course_id>/edit", methods=["GET", "POST"])
@admin_required
def update_course(course_id):
    course = Course.query.get_or_404(course_id)

    if request.method == "POST":
        course.title = request.form.get("title", course.title)
        course.description = request.form.get("description", course.description)
        course.video_url = request.form.get("video_url", course.video_url)

        video_file = request.files.get("video")
        if video_file and video_file.filename:
            filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
            path = os.path.join(app.root_path, 'static/learn', filename)
            video_file.save(path)
            course.video_filename = filename  # 存储文件名

        db.session.commit()
        flash("课程信息已更新")
        return redirect(url_for("course_detail", course_id=course_id))

    # 编辑页面使用
    levels = Level.query.filter_by(course_id=course_id).order_by(Level.level_number).all()
    with open("mlp/label_map_upper.json", "r", encoding="utf-8") as f:
        upper_map = json.load(f)
    with open("mlp/label_map_lower.json", "r", encoding="utf-8") as f:
        lower_map = json.load(f)
    return render_template("edit_course.html", course=course, levels=levels, upper_map=upper_map, lower_map=lower_map)

@app.route("/courses/<int:course_id>/levels/reorder", methods=["POST"])
@admin_required
def reorder_levels(course_id):
    try:
        data = request.get_json()
        ids = data.get("order", [])
        for i, level_id in enumerate(ids):
            level = Level.query.get(int(level_id))
            if level and level.course_id == course_id:
                level.level_number = i + 1
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/admin/edit_course/<int:course_id>', methods=['GET', 'POST'])
def edit_course(course_id):
    course = Course.query.get_or_404(course_id)

    if request.method == 'POST':
        course.title = request.form.get('title')
        course.description = request.form.get('description')

        video_file = request.files.get('video')
        if video_file and video_file.filename:
            filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
            path = os.path.join(app.root_path, 'static/learn', filename)
            video_file.save(path)
            course.video_filename = filename  # 替换原视频
        db.session.commit()

    return render_template('edit_course.html', course=course)

@app.route('/courses/<int:course_id>/delete_video', methods=['POST'])
@admin_required
def delete_course_video_admin(course_id):  # ✅ 改名避免冲突
    course = Course.query.get_or_404(course_id)
    if course.video_filename:
        path = os.path.join(app.root_path, 'static/learn', course.video_filename)
        if os.path.exists(path):
            os.remove(path)
        course.video_filename = None
        db.session.commit()
    flash("视频已删除")
    return redirect(url_for('update_course', course_id=course_id))



@app.route('/admin/delete_video/<int:course_id>', methods=['POST'])
def delete_course_video(course_id):
    course = Course.query.get_or_404(course_id)
    if course.video_filename:
        path = os.path.join(app.root_path, 'static/learn', course.video_filename)
        if os.path.exists(path):
            os.remove(path)
        course.video_filename = None
        db.session.commit()
    return redirect(url_for('edit_course', course_id=course.id))


# ------------------- 开始关卡 -------------------
@app.route("/learn/level/<int:level_id>")
@login_required
def start_level(level_id):
    level = Level.query.get_or_404(level_id)
    return render_template("do_level.html", level=level)

# ------------------- 实时识别接口 -------------------

@app.route("/api/verify_frame", methods=["POST"])
@login_required
def verify_frame():
    try:
        level_id = request.form.get("level_id")
        file = request.files.get("frame")
        if not file or not level_id:
            return jsonify({"success": False, "error": "缺少图像或关卡信息"}), 400

        # 解码图像
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 获取关卡信息
        level = Level.query.get_or_404(level_id)
        expected_upper = level.require_upper
        expected_lower = level.require_lower
        condition = level.pass_condition

        # 提取关键点
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        if not results.pose_landmarks:
            return jsonify({"success": True, "passed": False, "confidence": 0.0})

        keypoints = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]

        confidence = 0.0
        passed = False

        if condition == "upper":
            result = classify_confidence_from_keypoints(keypoints, expected_upper, part="upper")
            confidence = result["confidence"]
            passed = result["pass"]

        elif condition == "lower":
            result = classify_confidence_from_keypoints(keypoints, expected_lower, part="lower")
            confidence = result["confidence"]
            passed = result["pass"]

        elif condition == "both":
            result_upper = classify_confidence_from_keypoints(keypoints, expected_upper, part="upper")
            result_lower = classify_confidence_from_keypoints(keypoints, expected_lower, part="lower")
            confidence = min(result_upper["confidence"], result_lower["confidence"])
            passed = result_upper["pass"] and result_lower["pass"]

        # 更新通关记录
        if passed:
            user_id = session["user_id"]
            existing = Progress.query.filter_by(user_id=user_id, level_id=level_id).first()
            if not existing:
                db.session.add(Progress(user_id=user_id, level_id=level_id, passed=True))
            elif not existing.passed:
                existing.passed = True
            db.session.commit()

        return jsonify({
            "success": True,
            "passed": passed,
            "confidence": round(confidence, 4)  # 统一保留 4 位
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    
# 添加摄像头读取类（使用 OpenCV）
class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

camera = Camera()

def gen_frames(cam_index):
    cam = Camera(index=cam_index)
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route("/video_feed")
def video_feed():
    cam_index = int(request.args.get("cam", 0))
    return Response(gen_frames(cam_index), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)