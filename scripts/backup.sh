#!/bin/zsh
# NT-Sine 自动备份
#
# 策略（空间与安全平衡）：
#   每次运行   → 代码 + 文档 + 配置（约 1MB），文件名带分钟级时间戳
#   --full     → 追加模型权重 + 训练集 + 台词库（约 3GB），仅在训练/融合完成等关键节点执行
#   轮转       → 代码保留最近 120 份，完整备份保留最近 6 份；旧文件移入废纸篓而非直接删除
#
# 用法：
#   bash scripts/backup.sh          # 轻量备份（每次自动运行都做）
#   bash scripts/backup.sh --full   # 完整备份（关键节点）

SRC="/Users/ntars/Documents/Github/NT-Sine"
DEST="/Volumes/Expand/Backups/NT-Sine"
TS=$(date +%Y%m%d_%H%M%S)

# 备份卷未挂载时直接退出，不阻塞主流程
if [ ! -d "/Volumes/Expand" ]; then
  echo "[备份] 外置卷 /Volumes/Expand 未挂载，跳过本次备份"
  exit 0
fi

mkdir -p "$DEST/code" "$DEST/full"

# ── 轻量备份：代码与文档 ──
# 注意：--exclude 必须写在 -C 之前，写在文件列表之后会被 tar 当成文件名（实测踩过）
tar czf "$DEST/code/${TS}.tar.gz" \
  --exclude='.git' \
  --exclude='runs' \
  --exclude='models' \
  --exclude='data_train' \
  --exclude='data_factory/banks' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  -C "$SRC" . 2>/dev/null

CODE_SIZE=$(du -h "$DEST/code/${TS}.tar.gz" 2>/dev/null | cut -f1)
echo "[备份] 代码与文档：$DEST/code/${TS}.tar.gz（${CODE_SIZE}）"

# ── 完整备份：追加模型、数据、日志 ──
if [ "$1" = "--full" ]; then
  tar czf "$DEST/full/${TS}.tar.gz" \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    -C "$SRC" runs models data_train data_factory/banks 2>/dev/null
  FULL_SIZE=$(du -h "$DEST/full/${TS}.tar.gz" 2>/dev/null | cut -f1)
  echo "[备份] 完整备份：$DEST/full/${TS}.tar.gz（${FULL_SIZE}）"
fi

# ── 轮转：旧备份移入废纸篓，不做不可逆删除 ──
mkdir -p "$HOME/.Trash/nt-sine-backup"
ls -t "$DEST/code"/*.tar.gz 2>/dev/null | tail -n +121 | while read -r f; do
  mv "$f" "$HOME/.Trash/nt-sine-backup/" 2>/dev/null
done
ls -t "$DEST/full"/*.tar.gz 2>/dev/null | tail -n +7 | while read -r f; do
  mv "$f" "$HOME/.Trash/nt-sine-backup/" 2>/dev/null
done

echo "[备份] 当前占用：$(du -sh "$DEST" 2>/dev/null | cut -f1)｜代码 $(ls "$DEST/code" 2>/dev/null | wc -l | tr -d ' ') 份，完整 $(ls "$DEST/full" 2>/dev/null | wc -l | tr -d ' ') 份"
