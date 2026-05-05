#!/bin/bash

# 1. 定义临时打包目录
OUTPUT_DIR="dist_smart"
ZIP_NAME="baseline_ai.zip"

# 如果已存在则清理
rm -rf $OUTPUT_DIR
rm -f $ZIP_NAME
mkdir -p $OUTPUT_DIR

# 2. 复制核心入口文件
cp AI/main.py $OUTPUT_DIR/

# 3. 将新的 smart greedy 复制并重命名为 ai.py
cp AI/ai_baseline.py $OUTPUT_DIR/ai.py

# 4. 修复依赖丢失问题：复制 common.py
# (A) 放在根目录，适配 `from common import BaseAgent`
cp AI/common.py $OUTPUT_DIR/

# (B) 放在 AI/ 目录下，适配 `from AI.common import BaseAgent`
mkdir -p $OUTPUT_DIR/AI
cp AI/common.py $OUTPUT_DIR/AI/
# 确保 AI 被识别为一个 Python 包
touch $OUTPUT_DIR/AI/__init__.py

# 5. 复制必要的运行时依赖文件夹
cp -r SDK $OUTPUT_DIR/
cp -r tools $OUTPUT_DIR/

# 6. 执行压缩 (使用 Python 规避 zip 命令缺失)
cd $OUTPUT_DIR
python3 -m zipfile -c ../$ZIP_NAME .
cd ..

# 7. 清理临时目录
rm -rf $OUTPUT_DIR

echo "打包完成: $ZIP_NAME"