# tools/test_deepspeed.py
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="HF model to load (same as you use in unlearning)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto","fp16","bf16","fp32"])
    args = parser.parse_args()

    acc = Accelerator()
    rank = acc.process_index
    world = acc.num_processes

    # dtype
    if args.dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    if rank == 0:
        print(f"[test] world={world}, dtype={torch_dtype}, device={acc.device}")
        print(f"[test] using DeepSpeed: {acc.state.deepspeed_plugin is not None}")

    cfg = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = acc.prepare_model(model)

    # Direct (likely partial under ZeRO-3)
    try:
        sd_direct = model.state_dict()
        cnt_direct = len([k for k,v in sd_direct.items() if isinstance(v, torch.Tensor)])
    except Exception as e:
        cnt_direct = -1
        if rank == 0:
            print(f"[test] model.state_dict() failed: {e}")

    # Accelerate-gathered full state_dict
    try:
        sd_full = acc.get_state_dict(model)
        cnt_full = len([k for k,v in sd_full.items() if isinstance(v, torch.Tensor)])
    except Exception as e:
        cnt_full = -1
        if rank == 0:
            print(f"[test] accelerator.get_state_dict() failed: {e}")

    # Optional: Deepspeed ZeRO manual gather
    cnt_zero = -1
    try:
        import deepspeed
        m_unwrapped = acc.unwrap_model(model)
        params = list(m_unwrapped.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            cnt_zero = sum(1 for _ in m_unwrapped.named_parameters())
    except Exception:
        pass

    print(f"[rank{rank}] cnt_direct={cnt_direct}, cnt_full={cnt_full}, cnt_zeroGather={cnt_zero}")

if __name__ == "__main__":
    main()
