#!/usr/bin/env bash
set -euo pipefail

##########################################
# 0) 确保在仓库根目录执行（脚本在 scripts/ 下）
##########################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

##########################################
# 1) 只改这一行即可切换单/多卡
##########################################
export CUDA_VISIBLE_DEVICES=2         # 单卡: 2
# export CUDA_VISIBLE_DEVICES=2,3     # 双卡: 2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3 # 四卡: 0,1,2,3

##########################################
# 2) 清理分布式遗留变量 & 随机端口
##########################################
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export MASTER_PORT=$(python - <<'PY'
import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
echo "[INFO] Master Port: $MASTER_PORT"

##########################################
# 3) 根据 CUDA_VISIBLE_DEVICES 计算进程数
##########################################
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
  GPU_COUNT=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
fi
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "[ERR] GPU_COUNT < 1，检查 CUDA_VISIBLE_DEVICES"; exit 1
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> GPU_COUNT=$GPU_COUNT"

##########################################
# 4) 统一的 accelerate 启动器（覆盖进程数）
##########################################
LAUNCH=(accelerate launch --config_file configs/accelerate/default_config.yaml \
        --num_processes="$GPU_COUNT" --main_process_port "$MASTER_PORT")

##########################################
# 5) 数据资产（确保 idk.jsonl 在默认位置）
##########################################
IDK_PATH="$REPO_ROOT/data/idk.jsonl"
if [[ ! -f "$IDK_PATH" ]]; then
  echo "[WARN] $IDK_PATH 不存在。若跑 DPO + idk 配置需要该文件。"
  echo "       你可以执行:  mkdir -p data && curl -L https://huggingface.co/datasets/open-unlearning/idk/raw/main/idk.jsonl -o data/idk.jsonl"
fi

##########################################
# 6) 网格设置
##########################################
models=(
  "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
  "GradAscent unlearn/tofu/default.yaml"
  "GradDiff   unlearn/tofu/default.yaml"
  "NPO        unlearn/tofu/default.yaml"
  "DPO        unlearn/tofu/idk.yaml"
  "RMU        unlearn/tofu/default.yaml"
)
splits=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

# 训练超参
per_device_train_batch_size=4
gradient_accumulation_steps=4
# 如需“全局 batch size 不变”，可按需自动缩放 GA：
# base_gacc=4
# gradient_accumulation_steps=$(( base_gacc > GPU_COUNT ? base_gacc / GPU_COUNT : 1 ))

##########################################
# 7) 可执行文件绝对路径
##########################################
TRAIN_PY="$REPO_ROOT/src/train.py"
EVAL_PY="$REPO_ROOT/src/eval.py"

##########################################
# 8) 跳过/断点续跑逻辑：
#    - 若存在 evals/TOFU_SUMMARY.json：跳过训练与评测
#    - 若存在 checkpoint 或 trainer_state.json：跳过训练，仅评测
##########################################
for split in "${splits[@]}"; do
  forget_split=$(awk '{print $1}' <<< "$split")
  holdout_split=$(awk '{print $2}' <<< "$split")
  retain_split=$(awk '{print $3}' <<< "$split")

  for model in "${models[@]}"; do
    for trainer_experiment in "${trainers_experiments[@]}"; do
      trainer=$(awk '{print $1}' <<< "$trainer_experiment")
      experiment=$(awk '{print $2}' <<< "$trainer_experiment")

      task_name=tofu_${model}_${forget_split}_${trainer}
      run_dir="saves/unlearn/${task_name}"
      model_path=open-unlearning/tofu_${model}_full

      echo "[INFO] ${task_name}: Unlearning ${model_path} using ${trainer} (GPU_COUNT=${GPU_COUNT})"

      # --- Skip if eval already done
      if [[ -f "${run_dir}/evals/TOFU_SUMMARY.json" ]]; then
        echo "[SKIP] 已存在评测汇总 -> ${run_dir}/evals/TOFU_SUMMARY.json"
        continue
      fi

      need_train=1
      if [[ -f "${run_dir}/trainer_state.json" ]] || compgen -G "${run_dir}/checkpoint-* " >/dev/null; then
        need_train=0
        echo "[INFO] 检测到已有 checkpoint，执行 EVAL-only。"
      fi

      # --- Train (unless skipped)
      if [[ $need_train -eq 1 ]]; then
        "${LAUNCH[@]}" \
        "$TRAIN_PY" --config-name=unlearn.yaml \
          experiment=${experiment} \
          trainer=${trainer} \
          task_name=${task_name} \
          model=${model} \
          forget_split=${forget_split} \
          retain_split=${retain_split} \
          model.model_args.pretrained_model_name_or_path=${model_path} \
          retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
          trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
          trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
          trainer.args.ddp_find_unused_parameters=true \
          trainer.args.gradient_checkpointing=true
          # 说明：若你把 idk.jsonl 放在非默认路径，需要根据实际键名覆盖：
          # （见下文“键名确认”）
          # data.idk_path="$IDK_PATH"
      fi

      # --- Eval
      python "$EVAL_PY" \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${run_dir} \
        paths.output_dir=${run_dir}/evals \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

    done
  done
done





##########################################
# [POSTCHECK] 在脚本末尾追加：检测未完成评测并自动重训/重评
##########################################
echo "[POSTCHECK] Scanning for unfinished results..."

retrain_cnt=0
for split in "${splits[@]}"; do
  forget_split=$(awk '{print $1}' <<< "$split")
  holdout_split=$(awk '{print $2}' <<< "$split")
  retain_split=$(awk '{print $3}' <<< "$split")

  for model in "${models[@]}"; do
    for trainer_experiment in "${trainers_experiments[@]}"; do
      trainer=$(awk '{print $1}' <<< "$trainer_experiment")
      experiment=$(awk '{print $2}' <<< "$trainer_experiment")

      task_name=tofu_${model}_${forget_split}_${trainer}
      run_dir="saves/unlearn/${task_name}"
      eval_dir="${run_dir}/evals"
      summary_json="${eval_dir}/TOFU_SUMMARY.json"
      detail_json="${eval_dir}/TOFU_EVAL.json"
      summary_md="${eval_dir}/TOFU_SUMMARY.md"

      # 判定是否完成（Python 校验 JSON + 关键词；MD 作为补充）
      python - <<'PY' "$summary_json" "$detail_json" "$summary_md"
import json,sys,os
sj,dj,sm = sys.argv[1:4]
def ok_json(p):
    try:
        if not (os.path.exists(p) and os.path.getsize(p) > 20): return False
        with open(p,'r') as f: j=json.load(f)
        s=json.dumps(j)
        req = ("forget_quality","model_utility","privleak","extraction_strength")
        return all(k in s for k in req)
    except Exception:
        return False
def ok_md(p):
    try:
        if not (os.path.exists(p) and os.path.getsize(p) > 50): return False
        t=open(p,encoding="utf-8",errors="ignore").read().lower()
        keys=("forget_quality","model_utility","privleak","extraction_strength")
        return sum(k in t for k in keys) >= 2
    except Exception:
        return False
ok = (ok_json(sj) and ok_json(dj)) or ok_md(sm)
sys.exit(0 if ok else 1)
PY
      status=$?

      if [[ $status -ne 0 ]]; then
        echo "[POSTCHECK][RE-RUN] ${task_name} seems unfinished. Retrain + re-eval..."

        # 重新训练（带断点续训）
        accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
          src/train.py --config-name=unlearn.yaml \
          experiment=${experiment} \
          trainer=${trainer} \
          task_name=${task_name} \
          model=${model} \
          forget_split=${forget_split} \
          retain_split=${retain_split} \
          model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
          retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
          trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
          trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
          trainer.args.ddp_find_unused_parameters=true \
          trainer.args.gradient_checkpointing=true \
          trainer.args.resume_from_checkpoint=last

        # 重新评测
        mkdir -p "${eval_dir}"
        python src/eval.py \
          experiment=eval/tofu/default.yaml \
          forget_split=${forget_split} \
          holdout_split=${holdout_split} \
          model=${model} \
          task_name=${task_name} \
          model.model_args.pretrained_model_name_or_path=${run_dir} \
          paths.output_dir=${eval_dir} \
          retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

        retrain_cnt=$((retrain_cnt+1))
        # 二次校验（仅提示，不再循环重试）
        python - <<'PY' "$summary_json" "$detail_json" "$summary_md"
import json,sys,os
sj,dj,sm = sys.argv[1:4]
def ok(p):
    try:
        if not os.path.exists(p) or os.path.getsize(p) <= 20: return False
        json.load(open(p))
        return True
    except Exception:
        return False
print("[POSTCHECK] Re-run status:",
      "OK" if (ok(sj) and ok(dj)) else "STILL_INCOMPLETE")
PY
      else
        echo "[POSTCHECK][OK] ${task_name} is complete."
      fi
    done
  done
done

echo "[POSTCHECK] Re-runs triggered: ${retrain_cnt}"

