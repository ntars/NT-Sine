#!/bin/zsh
# 官网部署 —— 只把「仓库追踪的文件」传上线
#
# 为什么不用项目根直接部署：EdgeOne CLI 会把整个目录打包上传，
# 项目根含 models/（2.1G 融合权重）、runs/ 与 data_train/，
# 实测生成 2.3G 的 .edgeone/assets，既慢又会把训练数据送上 CDN。
# 用 git archive 导出干净副本（约 1MB），单一真源仍是仓库本身。
#
# 用法：bash scripts/deploy_site.sh
# 输出最后一行是部署结果 JSON，url 字段为完整访问地址（含 eo_token 查询串）。

set -e
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:-nt-sine-site}
STAGE=$(mktemp -d /tmp/nt-sine-site-stage.XXXXXX)

# git archive 只导出已提交内容，天然排除 gitignore 里的权重、数据集与 .env
git archive HEAD | tar -x -C "$STAGE"

cd "$STAGE"
RESULT=$(PAGES_SOURCE=skills edgeone makers deploy -n "$PROJECT" --json 2>&1 | tail -1)
echo "$RESULT"

# 暂存目录走废纸篓，不做不可逆删除
mv "$STAGE" "$HOME/.Trash/nt-sine-site-stage-$(date +%H%M%S)" 2>/dev/null || true
