#!/usr/bin/env bash
# 获取参考评分器（M² scorer 与 GLEU），固定到本文使用的提交版本；它们保留各自的许可证，不随本仓库分发。
set -e
cd "$(dirname "$0")/../.."
mkdir -p vendor
if [ ! -d vendor/Standard_Korean_GEC ]; then
  git clone https://github.com/soyoung97/Standard_Korean_GEC.git vendor/Standard_Korean_GEC
fi
git -C vendor/Standard_Korean_GEC checkout dfe0af9db60078edf9e35400d1a9e96148846050
echo "vendor ready: $(git -C vendor/Standard_Korean_GEC log -1 --format=%H)"
