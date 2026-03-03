# 使用官方 Python 3.10 镜像 (MediaPipe 兼容性较好)
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (OpenCV 需要)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 使用 --no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要的目录 (用于存储上传文件和数据库)
RUN mkdir -p static/learn instance

# 暴露端口
EXPOSE 8000

# 启动命令 (使用 gunicorn 生产级服务器)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]
