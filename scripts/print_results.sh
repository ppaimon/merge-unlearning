#!/usr/bin/env bash
set -euo pipefail

# == 进入仓库根、确定输出路径（与脚本同目录） ==
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_MD="${SCRIPT_DIR}/TOFU_RUNS_SUMMARY.md"

# == 收集结果文件 ==
shopt -s nullglob
RESULTS=(saves/unlearn/*/evals/TOFU_SUMMARY.json)

# == 写入 Markdown 头 ==
{
  echo "# TOFU Experiment Summary"
  echo
  echo "- Generated at: \`$(date "+%Y-%m-%d %H:%M:%S %Z")\`"
  echo "- Repo root: \`${REPO_ROOT}\`"
  echo
  echo "| Task | Model | Method | Forget | Holdout | Retain | Forget Quality | Forget Truth Ratio | Model Utility | Privleak | Extraction Strength |"
  echo "|---|---|---|---|---|---|---:|---:|---:|---:|---:|"

  if [[ ${#RESULTS[@]} -eq 0 ]]; then
    echo "| _No results found_ |  |  |  |  |  |  |  |  |  |  |"
  else
    for SUM in "${RESULTS[@]}"; do
      EVAL_DIR="$(dirname "$SUM")"
      RUN_DIR="$(dirname "$EVAL_DIR")"
      TASK_NAME="$(basename "$RUN_DIR")"

      # 解析 task_name = tofu_${model}_${forget_split}_${trainer}
      core="${TASK_NAME#tofu_}"     # ${model}_${forget_split}_${trainer}
      METHOD="${core##*_}"
      tmp="${core%_*}"              # ${model}_${forget_split}
      FORGET_SPLIT="${tmp##*_}"
      MODEL="${tmp%_*}"

      # 用 Python 读取 JSON 指标与 holdout/retain 线索（若缺失则留空）
      PYOUT="$(
        python - <<'PY' "$SUM"
import json,sys,os,re
sj=sys.argv[1]

def jload(p):
    try:
        with open(p,'r',encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

s=jload(sj)
# 常见聚合指标
fq = s.get("forget_quality")
# 不同版本可能有不同命名
ftr = s.get("forget_Truth_Ratio", s.get("forget_truth_ratio", None))
mu = s.get("model_utility")
pl = s.get("privleak")
es = s.get("extraction_strength")

dj=os.path.join(os.path.dirname(sj),"TOFU_EVAL.json")
d=jload(dj)

def find_first(key, obj):
    if isinstance(obj, dict):
        for k,v in obj.items():
            if k==key and isinstance(v,str):
                return v
            r=find_first(key,v)
            if r: return r
    elif isinstance(obj, list):
        for v in obj:
            r=find_first(key,v)
            if r: return r
    return ""

hs = find_first("holdout_split", s) or find_first("holdout_split", d) or ""

def find_retain(obj):
    text=json.dumps(obj,ensure_ascii=False)
    m=re.search(r"retain(\d+)", text)
    return f"retain{m.group(1)}" if m else ""
rs = find_retain(s) or find_retain(d) or ""

def tostr(x):
    return "" if x is None else str(x)

print("\t".join(map(tostr, [fq, ftr, mu, pl, es, rs, hs])))
PY
      )"

      IFS=$'\t' read -r FQ FTR MU PRIV ES RETAIN_SPLIT HOLDOUT_SPLIT <<<"$PYOUT"

      # 输出一行 Markdown
      printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$TASK_NAME" "$MODEL" "$METHOD" "$FORGET_SPLIT" \
        "${HOLDOUT_SPLIT:-}" "${RETAIN_SPLIT:-}" \
        "${FQ:-}" "${FTR:-}" "${MU:-}" "${PRIV:-}" "${ES:-}"
    done
  fi
} > "$OUTPUT_MD"

echo "[OK] Summary written to: $OUTPUT_MD"
