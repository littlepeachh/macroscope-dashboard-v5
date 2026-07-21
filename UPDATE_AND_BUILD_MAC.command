#!/bin/bash
set -u
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "请先运行 START_LOCAL_MAC.command"; read -r; exit 1; }
python scripts/update_data.py
python scripts/build_site.py
echo "更新完成。刷新浏览器即可。"
read -r -p "按回车关闭..."
