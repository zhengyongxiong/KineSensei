#!/bin/bash

# 检查是否已接受 Xcode 许可
if ! git --version >/dev/null 2>&1; then
    echo "❌ Git 无法运行。请先运行 'sudo xcodebuild -license' 接受许可协议。"
    exit 1
fi

echo "🚀 开始初始化 Git 仓库..."

# 初始化
if [ ! -d ".git" ]; then
    git init
    git branch -M main
    git remote add origin https://github.com/zhengyongxiong/KineSensei.git
    echo "✅ 仓库已初始化"
else
    echo "ℹ️  仓库已存在，跳过初始化"
fi

# 添加文件
git add .

# 提交
git commit -m "Initial commit with project documentation and structure"

# 推送
echo "📤 正在推送到 GitHub..."
git push -u origin main

echo "🎉 完成！"
