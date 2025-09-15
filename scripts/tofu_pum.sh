#!/usr/bin/env bash
set -euo pipefail

##########################################
# Tuning Hyperparameters (centralized block)
# - Defaults are sensible; ranges are recommended search windows.
# - Edit here or override via environment variables before running.
##########################################
# Training throughput
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-2}       # per-device batch size; range: 1–8 (depends on VRAM)
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}          # gradient accumulation; range: 1–16 (match global batch)

# PUM outer loop
PUM_COPIES_M=${PUM_COPIES_M:-2}                  # number of perturbed copies per round (m); range: 4–16
PUM_ROUNDS_R=${PUM_ROUNDS_R:-1}                  # number of PUM rounds (R); range: 1–3
PUM_ALPHA_MIN=${PUM_ALPHA_MIN:-1.0}              # secret scaling min α_min (≥1); range: 1.0–1.5
PUM_ALPHA_MAX=${PUM_ALPHA_MAX:-1.1}              # secret scaling max α_max; range: α_min–1.5
PUM_ETA_SRV=${PUM_ETA_SRV:-1.0}                  # server step size η_srv; range: 0.5–1.5
PUM_THETA_REF_BETA=${PUM_THETA_REF_BETA:-0.8}    # EMA β for DP clipping reference; range: 0.7–0.9
PUM_JITTER_TAU=${PUM_JITTER_TAU:-0.0}            # tiny jitter τ per copy (std); range: 0–1e-3 (0 disables)
PUM_USE_REPARAM=${PUM_USE_REPARAM:-true}        # per-copy orthogonal/permutation reparameterization; recommend true if stable

# Local unlearning budget (per copy)
PUM_LOCAL_EPOCHS=${PUM_LOCAL_EPOCHS:-1}          # local epochs per copy; range: 1–3
PUM_LOCAL_MAX_STEPS=${PUM_LOCAL_MAX_STEPS:-null} # optional steps cap per copy; null for auto-balance

# Optional robustness (server-side clipping of client updates)
PUM_CLIP_UPDATE_NORM=${PUM_CLIP_UPDATE_NORM:-null}               # global L2 clip on aligned update; e.g., 5.0–20.0
PUM_CLIP_UPDATE_PER_LAYER=${PUM_CLIP_UPDATE_PER_LAYER:-null}     # per-layer L2 clips; e.g., "[5,5,...]"

# DP calibration and publication
PUM_PER_LAYER_NOISE=${PUM_PER_LAYER_NOISE:-true} # true: calibrate per-layer σ_ℓ; recommended when using DP
DP_EPSILON=${DP_EPSILON:-8}                    # target ε; range: 2–8 (dataset/task-dependent)
DP_DELTA=${DP_DELTA:-0.000025}                        # target δ; e.g., 2.5e-5 for TOFU (1/(10N)); range: 1e-8–1e-5
DP_RDP_ORDERS=${DP_RDP_ORDERS:-[1.5,2,3,4,8,16,32,64,128]} # RDP orders sweep
DP_PER_LAYER_ALLOC=${DP_PER_LAYER_ALLOC:-auto}    # noise allocation across layers: auto|min-var, equalized, varmin
DP_SENS_TOTAL_L2=${DP_SENS_TOTAL_L2:-null}       # total sensitivity Δ̄_2 (≈ 2 C_total); used for single-σ or fallback
DP_SENS_PER_LAYER_L2=${DP_SENS_PER_LAYER_L2:-null} # per-layer sensitivity list (≈ 2 C_ℓ); preferred for per-layer σ
DP_USE_WORSTCASE_ALPHA=${DP_USE_WORSTCASE_ALPHA:-true} # DP bound uses worst-case S_α (recommended true)

# Optional manual noise (discouraged; DP overrides when provided)
PUM_SIGMA=${PUM_SIGMA:-0.0}                       # global σ if DP not set
PUM_SIGMA_PER_LAYER=${PUM_SIGMA_PER_LAYER:-null}  # per-layer σ list if DP not set; e.g., "[0.02,0.02,...]"

# Server-side DP clipping of publication center (θ~)
PUM_SERVER_CENTER_CLIP=${PUM_SERVER_CENTER_CLIP:-null}          # null:auto (enable if DP sensitivities/C provided); true/false
PUM_CENTER_CLIP_C_GLOBAL=${PUM_CENTER_CLIP_C_GLOBAL:-null}      # optional global C for non-layer params (Δ̄_2=2C)
PUM_CENTER_CLIP_C_PER_LAYER=${PUM_CENTER_CLIP_C_PER_LAYER:-null}


PUM_CENTER_CLIP_Q=${PUM_CENTER_CLIP_Q:-0.95}                    # quantile q for C_ℓ (0.90–0.99)
PUM_CENTER_CLIP_KAPPA=${PUM_CENTER_CLIP_KAPPA:-1.25}            # scaling κ for C_ℓ (1.1–1.5)
PUM_CENTER_CLIP_GAMMA=${PUM_CENTER_CLIP_GAMMA:-1.3}             # round scaling γ for multi-round
PUM_CENTER_CLIP_REF_MODELS=${PUM_CENTER_CLIP_REF_MODELS:-null}  # optional list of ref model paths (e.g., "[/path/a,/path/b]")

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
per_device_train_batch_size=${BATCH_SIZE_PER_GPU}
gradient_accumulation_steps=${GRAD_ACCUM_STEPS}

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
# 7) 组合设置：模型 × (PUM(inner), 实验配置) × 数据 split
##########################################
# 模型（默认启用 8B；按需打开其它）
MODELS=(
  "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
)

# PUM 的内层方法 + 实验配置对
# 说明：外层 trainer 固定为 PUM；这里配置 inner_handler 与实验配置
PUM_INNERS_EXPERIMENTS=(
  # "GradAscent unlearn/tofu/default.yaml"
  # "GradDiff   unlearn/tofu/default.yaml"
  "NPO        unlearn/tofu/default.yaml"
  # "DPO        unlearn/tofu/idk.yaml"
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
# 10) 主循环：训练 + 评测（PUM）
##########################################
for split in "${SPLITS[@]}"; do
  forget_split=$(awk '{print $1}' <<< "$split")
  holdout_split=$(awk '{print $2}' <<< "$split")
  retain_split=$(awk '{print $3}' <<< "$split")

  for model in "${MODELS[@]}"; do
    for te in "${PUM_INNERS_EXPERIMENTS[@]}"; do
      inner=$(awk '{print $1}' <<< "$te")
      experiment=$(awk '{print $2}' <<< "$te")

      trainer="PUM"
      task_name="tofu_${model}_${forget_split}_PUM-${inner}"
      run_dir="saves/unlearn/${task_name}"
      model_path="open-unlearning/tofu_${model}_full"

      echo "[INFO] ${task_name}: Unlearning ${model_path} via PUM(inner=${inner}) (GPUs=${GPU_COUNT})"

      # 若 RERUN=True，删除旧目录以确保从零开始
      if [[ $RERUN_FLAG -eq 1 ]]; then
        if [[ -d "$run_dir" ]]; then
          echo "[RERUN] Removing existing run_dir: $run_dir"
          rm -rf "$run_dir"
        fi
      fi

      # 是否跳过训练与评测
      if [[ $RERUN_FLAG -eq 0 ]]; then
        if [[ -f "${run_dir}/evals/TOFU_SUMMARY.json" ]]; then
          echo "[SKIP] 已存在评测汇总 -> ${run_dir}/evals/TOFU_SUMMARY.json"
          continue
        fi
      fi

      ########################
      # 训练（PUM）
      ########################
      need_train=1
      if [[ $RERUN_FLAG -eq 0 ]]; then
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
          trainer.args.gradient_checkpointing=true \
          trainer.method_args.inner_handler="${inner}" \
          trainer.method_args.copies_m="${PUM_COPIES_M}" \
          trainer.method_args.rounds_R="${PUM_ROUNDS_R}" \
          trainer.method_args.sigma="${PUM_SIGMA}" \
          trainer.method_args.sigma_per_layer="${PUM_SIGMA_PER_LAYER}" \
          trainer.method_args.per_layer_noise="${PUM_PER_LAYER_NOISE}" \
          trainer.method_args.dp_epsilon="${DP_EPSILON}" \
          trainer.method_args.dp_delta="${DP_DELTA}" \
          trainer.method_args.dp_sensitivity_total_l2="${DP_SENS_TOTAL_L2}" \
          trainer.method_args.dp_sensitivity_per_layer_l2="${DP_SENS_PER_LAYER_L2}" \
          trainer.method_args.dp_rdp_orders="${DP_RDP_ORDERS}" \
          trainer.method_args.dp_use_worstcase_alpha="${DP_USE_WORSTCASE_ALPHA}" \
          trainer.method_args.dp_per_layer_allocation="${DP_PER_LAYER_ALLOC}" \
          trainer.method_args.alpha_min="${PUM_ALPHA_MIN}" \
          trainer.method_args.alpha_max="${PUM_ALPHA_MAX}" \
          trainer.method_args.eta_srv="${PUM_ETA_SRV}" \
          trainer.method_args.theta_ref_beta="${PUM_THETA_REF_BETA}" \
          trainer.method_args.server_center_clipping="${PUM_SERVER_CENTER_CLIP}" \
          trainer.method_args.center_clip_C_global="${PUM_CENTER_CLIP_C_GLOBAL}" \
          trainer.method_args.center_clip_C_per_layer="${PUM_CENTER_CLIP_C_PER_LAYER}" \
          trainer.method_args.center_clip_quantile_q="${PUM_CENTER_CLIP_Q}" \
          trainer.method_args.center_clip_quantile_kappa="${PUM_CENTER_CLIP_KAPPA}" \
          trainer.method_args.center_clip_round_gamma="${PUM_CENTER_CLIP_GAMMA}" \
          trainer.method_args.center_clip_ref_model_paths="${PUM_CENTER_CLIP_REF_MODELS}" \
          trainer.method_args.jitter_tau="${PUM_JITTER_TAU}" \
          trainer.method_args.local_epochs="${PUM_LOCAL_EPOCHS}" \
          trainer.method_args.local_max_steps="${PUM_LOCAL_MAX_STEPS}" \
          trainer.method_args.auto_balance_local_max_steps=true \
          trainer.method_args.clip_update_norm="${PUM_CLIP_UPDATE_NORM}" \
          trainer.method_args.clip_update_norm_per_layer="${PUM_CLIP_UPDATE_PER_LAYER}" \
          trainer.method_args.use_orthogonal_reparam="${PUM_USE_REPARAM}"
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
      for te in "${PUM_INNERS_EXPERIMENTS[@]}"; do
        inner=$(awk '{print $1}' <<< "$te")
        experiment=$(awk '{print $2}' <<< "$te")

        task_name="tofu_${model}_${forget_split}_PUM-${inner}"
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
            trainer=PUM \
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
          trainer.args.resume_from_checkpoint=last \
          trainer.method_args.inner_handler="${inner}" \
          trainer.method_args.copies_m="${PUM_COPIES_M}" \
          trainer.method_args.rounds_R="${PUM_ROUNDS_R}" \
          trainer.method_args.sigma="${PUM_SIGMA}" \
          trainer.method_args.sigma_per_layer="${PUM_SIGMA_PER_LAYER}" \
          trainer.method_args.per_layer_noise="${PUM_PER_LAYER_NOISE}" \
          trainer.method_args.dp_epsilon="${DP_EPSILON}" \
          trainer.method_args.dp_delta="${DP_DELTA}" \
          trainer.method_args.dp_sensitivity_total_l2="${DP_SENS_TOTAL_L2}" \
          trainer.method_args.dp_sensitivity_per_layer_l2="${DP_SENS_PER_LAYER_L2}" \
          trainer.method_args.dp_rdp_orders="${DP_RDP_ORDERS}" \
          trainer.method_args.dp_use_worstcase_alpha="${DP_USE_WORSTCASE_ALPHA}" \
          trainer.method_args.dp_per_layer_allocation="${DP_PER_LAYER_ALLOC}" \
          trainer.method_args.alpha_min="${PUM_ALPHA_MIN}" \
          trainer.method_args.alpha_max="${PUM_ALPHA_MAX}" \
          trainer.method_args.eta_srv="${PUM_ETA_SRV}" \
          trainer.method_args.theta_ref_beta="${PUM_THETA_REF_BETA}" \
          trainer.method_args.server_center_clipping="${PUM_SERVER_CENTER_CLIP}" \
          trainer.method_args.center_clip_C_global="${PUM_CENTER_CLIP_C_GLOBAL}" \
          trainer.method_args.center_clip_C_per_layer="${PUM_CENTER_CLIP_C_PER_LAYER}" \
          trainer.method_args.center_clip_quantile_q="${PUM_CENTER_CLIP_Q}" \
          trainer.method_args.center_clip_quantile_kappa="${PUM_CENTER_CLIP_KAPPA}" \
          trainer.method_args.center_clip_round_gamma="${PUM_CENTER_CLIP_GAMMA}" \
          trainer.method_args.center_clip_ref_model_paths="${PUM_CENTER_CLIP_REF_MODELS}" \
          trainer.method_args.jitter_tau="${PUM_JITTER_TAU}" \
          trainer.method_args.local_epochs="${PUM_LOCAL_EPOCHS}" \
          trainer.method_args.local_max_steps="${PUM_LOCAL_MAX_STEPS}" \
          trainer.method_args.auto_balance_local_max_steps=true \
          trainer.method_args.clip_update_norm="${PUM_CLIP_UPDATE_NORM}" \
          trainer.method_args.clip_update_norm_per_layer="${PUM_CLIP_UPDATE_PER_LAYER}" \
          trainer.method_args.use_orthogonal_reparam="${PUM_USE_REPARAM}"

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
