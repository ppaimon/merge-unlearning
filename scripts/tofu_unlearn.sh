#!/usr/bin/env bash
set -euo pipefail

##########################################
# 0) 在仓库根目录执行（脚本位于 scripts/ 下）
##########################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# 推荐的稳定性/性能相关环境变量（单机多卡）
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

##########################################
# 1) 设备可见性：按需修改为你的 4 张卡
#    亦可通过 `GPUS` 环境变量覆盖
##########################################
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"

##########################################
# 8) 训练超参（可按显存/吞吐调节）
##########################################
per_device_train_batch_size=${BATCH_SIZE_PER_GPU:-2}
gradient_accumulation_steps=${GRAD_ACCUM_STEPS:-4}


##########################################
# 2) RERUN 开关：True=完全重跑（删除输出目录）
##########################################
RERUN=True

RERUN_FLAG=0
case "${RERUN:-false}" in
  [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) RERUN_FLAG=1;;
  *) RERUN_FLAG=0;;
esac
echo "[INFO] RERUN=$RERUN_FLAG (1=full rerun, 0=skip if done)"

##########################################
# 3) 清理分布式遗留变量 & 随机端口
##########################################
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$(
python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0))
print(s.getsockname()[1]); s.close()
PY
)"
echo "[INFO] Master Port: $MASTER_PORT"

##########################################
# 4) 根据 CUDA_VISIBLE_DEVICES 计算并发进程数
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
# 5) 统一的 accelerate 启动器（覆盖进程数）
##########################################
LAUNCH=(
  accelerate launch
  --config_file configs/accelerate/default_config.yaml
  --num_processes "$GPU_COUNT"
  --main_process_port "$MASTER_PORT"
)

##########################################
# 6) DPO 所需数据资产（如启用 idk 配置）
##########################################
IDK_PATH="$REPO_ROOT/data/idk.jsonl"
if [[ ! -f "$IDK_PATH" ]]; then
  echo "[WARN] $IDK_PATH 不存在。若运行 DPO + idk 配置需要该文件。"
  echo "  可执行：mkdir -p data && curl -L https://huggingface.co/datasets/open-unlearning/idk/raw/main/idk.jsonl -o data/idk.jsonl"
fi

##########################################
# 7) 组合设置：模型 × (方法, 实验配置) × 数据 split
##########################################
# 模型（默认仅启用 3B；如需 1B/8B 打开注释）
MODELS=(
  "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
)

# 方法-实验配置对（DPO 使用 unlearn/tofu/idk.yaml）
TRAINERS_EXPERIMENTS=(
  # "GradAscent unlearn/tofu/default.yaml"
  # "GradDiff   unlearn/tofu/default.yaml"
  "NPO        unlearn/tofu/default.yaml"
  # "DPO        unlearn/tofu/idk.yaml"
  # "RMU        unlearn/tofu/default.yaml"
)

# 数据 split（forget/holdout/retain）
SPLITS=(
  "forget01 holdout01 retain99"
  # "forget05 holdout05 retain95"
  # "forget10 holdout10 retain90"
)


##########################################
# 9) 可执行脚本
##########################################
TRAIN_PY="$REPO_ROOT/src/train.py"
EVAL_PY="$REPO_ROOT/src/eval.py"

##########################################
# 10) 主循环：训练 + 评测
##########################################
for split in "${SPLITS[@]}"; do
  forget_split=$(awk '{print $1}' <<< "$split")
  holdout_split=$(awk '{print $2}' <<< "$split")
  retain_split=$(awk '{print $3}' <<< "$split")

  for model in "${MODELS[@]}"; do
    for te in "${TRAINERS_EXPERIMENTS[@]}"; do
      trainer=$(awk '{print $1}' <<< "$te")
      experiment=$(awk '{print $2}' <<< "$te")

      task_name="tofu_${model}_${forget_split}_${trainer}"
      run_dir="saves/unlearn/${task_name}"
      model_path="open-unlearning/tofu_${model}_full"

      echo "[INFO] ${task_name}: Unlearning ${model_path} via ${trainer} (GPUs=${GPU_COUNT})"

      # 若 RERUN=True，删除旧目录以确保从零开始
      if [[ $RERUN_FLAG -eq 1 ]]; then
        if [[ -d "$run_dir" ]]; then
          echo "[RERUN] Removing existing run_dir: $run_dir"
          rm -rf "$run_dir"
        fi
      fi

      # 是否跳过训练与评测
      # - RERUN=False：若已有产物则跳过
      # - RERUN=True ：不跳过（强制重跑）
      if [[ $RERUN_FLAG -eq 0 ]]; then
        if [[ -f "${run_dir}/evals/TOFU_SUMMARY.json" ]]; then
          echo "[SKIP] 已存在评测汇总 -> ${run_dir}/evals/TOFU_SUMMARY.json"
          continue
        fi
      fi

      ########################
      # 训练
      ########################
      need_train=1
      if [[ $RERUN_FLAG -eq 0 ]]; then
        # 仅在非 RERUN 模式下允许“评测-only”
        if [[ -f "${run_dir}/trainer_state.json" ]] || compgen -G "${run_dir}/checkpoint-*" >/dev/null; then
          need_train=0
          echo "[INFO] 检测到已有 checkpoint，执行 EVAL-only。"
        fi
      fi

      if [[ $need_train -eq 1 ]]; then
        "${LAUNCH[@]}" \
          "$TRAIN_PY" --config-name=unlearn.yaml \
          experiment="${experiment}" \
          trainer="${trainer}" \
          task_name="${task_name}" \
          model="${model}" \
          forget_split="${forget_split}" \
          retain_split="${retain_split}" \
          model.model_args.pretrained_model_name_or_path="${model_path}" \
          retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json" \
          trainer.args.per_device_train_batch_size="${per_device_train_batch_size}" \
          trainer.args.gradient_accumulation_steps="${gradient_accumulation_steps}" \
          trainer.args.ddp_find_unused_parameters=true \
          trainer.args.gradient_checkpointing=true
          # 如你的 DPO 代码需显式指定 idk 路径，可解注下一行
          # data.idk_path="${IDK_PATH}"
      fi

      ########################
      # 评测
      ########################
      python "$EVAL_PY" \
        experiment=eval/tofu/default.yaml \
        forget_split="${forget_split}" \
        holdout_split="${holdout_split}" \
        model="${model}" \
        task_name="${task_name}" \
        model.model_args.pretrained_model_name_or_path="${run_dir}" \
        paths.output_dir="${run_dir}/evals" \
        retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

    done
  done
done

##########################################
# 11) 可选的补全检查（非 RERUN 时启用）
##########################################
if [[ $RERUN_FLAG -eq 0 ]]; then
  echo "[POSTCHECK] Scanning for unfinished results..."
  retrain_cnt=0

  for split in "${SPLITS[@]}"; do
    forget_split=$(awk '{print $1}' <<< "$split")
    holdout_split=$(awk '{print $2}' <<< "$split")
    retain_split=$(awk '{print $3}' <<< "$split")

    for model in "${MODELS[@]}"; do
      for te in "${TRAINERS_EXPERIMENTS[@]}"; do
        trainer=$(awk '{print $1}' <<< "$te")
        experiment=$(awk '{print $2}' <<< "$te")

        task_name="tofu_${model}_${forget_split}_${trainer}"
        run_dir="saves/unlearn/${task_name}"
        eval_dir="${run_dir}/evals"

        summary_json="${eval_dir}/TOFU_SUMMARY.json"
        detail_json="${eval_dir}/TOFU_EVAL.json"
        summary_md="${eval_dir}/TOFU_SUMMARY.md"

        if ! python - "$summary_json" "$detail_json" "$summary_md" <<'PY'
import json,sys,os
sj,dj,sm = sys.argv[1:4]

def ok_json(p):
    try:
        if not (os.path.exists(p) and os.path.getsize(p) > 20):
            return False
        j=json.load(open(p,'r'))
        s=json.dumps(j)
        req=("forget_quality","model_utility","privleak","extraction_strength")
        return all(k in s for k in req)
    except Exception:
        return False

def ok_md(p):
    try:
        if not (os.path.exists(p) and os.path.getsize(p) > 50):
            return False
        t=open(p,encoding="utf-8",errors="ignore").read().lower()
        keys=("forget_quality","model_utility","privleak","extraction_strength")
        return sum(k in t for k in keys) >= 2
    except Exception:
        return False

ok = (ok_json(sj) and ok_json(dj)) or ok_md(sm)
sys.exit(0 if ok else 1)
PY
        then
          echo "[POSTCHECK][RE-RUN] ${task_name} seems unfinished. Retrain + re-eval..."

          "${LAUNCH[@]}" \
            "$TRAIN_PY" --config-name=unlearn.yaml \
            experiment="${experiment}" \
            trainer="${trainer}" \
            task_name="${task_name}" \
            model="${model}" \
            forget_split="${forget_split}" \
            retain_split="${retain_split}" \
            model.model_args.pretrained_model_name_or_path="open-unlearning/tofu_${model}_full" \
            retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json" \
            trainer.args.per_device_train_batch_size="${per_device_train_batch_size}" \
            trainer.args.gradient_accumulation_steps="${gradient_accumulation_steps}" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.resume_from_checkpoint=last

          python "$EVAL_PY" \
            experiment=eval/tofu/default.yaml \
            forget_split="${forget_split}" \
            holdout_split="${holdout_split}" \
            model="${model}" \
            task_name="${task_name}" \
            model.model_args.pretrained_model_name_or_path="${run_dir}" \
            paths.output_dir="${eval_dir}" \
            retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

          retrain_cnt=$((retrain_cnt+1))
        else
          echo "[POSTCHECK][OK] ${task_name} is complete."
        fi
      done
    done
  done

  echo "[POSTCHECK] Re-runs triggered: ${retrain_cnt}"
fi
