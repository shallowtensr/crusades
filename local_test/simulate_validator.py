"""
Local Validator Simulation Tool.

Use this script to test your train.py exactly as the production validator does.

1. Build the docker image (run from repo root):
   docker build --network=host -f environments/templar/Dockerfile --no-cache -t templar-eval:latest .

2. Run the simulation:
   docker run --gpus all -it --rm \
     -v $(pwd)/local_test/train.py:/test/train.py \
     -v $(pwd)/local_test/simulate_validator.py:/test/simulate.py \
     -v $(pwd)/hparams/hparams.json:/app/hparams.json \
     -e PYTHONPATH=/app \
     templar-eval:latest \
     python3 /test/simulate.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the paths required to import the validator's environment
# Use /app for the image's env.py and stubs
sys.path.append("/app")

from env import Actor

async def simulate():
    print("🚀 Starting Simulated Validator Test...")
    
    # 1. Load HParams (from the image's path or mounted path)
    hparams_path = Path("/app/hparams.json") 
    if not hparams_path.exists():
        hparams_path = Path("/app/hparams/hparams.json")
    
    if not hparams_path.exists():
        print(f"ERROR: hparams.json not found at /app/hparams.json or /app/hparams/hparams.json")
        print("Launch it with the command at the top of this file")
        sys.exit(1)

    print(f"Loading hparams from: {hparams_path}")
    with open(hparams_path) as f:
        hb = json.load(f)
    
    # 2. Load your WIP code
    # We will mount this to /test/train.py
    code_path = Path("/test/train.py")
    if not code_path.exists() or not code_path.is_file():
        print(f"ERROR: train.py not found at {code_path}")
        print("Launch it with the command at the top of this file")
        sys.exit(1)

    print(f"Loading miner code from: {code_path}")
    code = code_path.read_text()
    
    # 3. Setup Actor
    actor = Actor()
    
    # 4. Run Evaluation (mirrors the validator's thresholds from hparams)
    result = await actor.evaluate(
        task_id=1337,
        seed="local:test:1",
        model_url=hb["benchmark_model_name"],
        data_url=hb["benchmark_dataset_name"],
        steps=hb["eval_steps"],
        batch_size=hb["benchmark_batch_size"],
        sequence_length=hb["benchmark_sequence_length"],
        code=code,
        # Pass all the verification thresholds from hparams
        max_loss_difference=hb["verification"]["max_loss_difference"],
        gradient_norm_ratio_max=hb["verification"]["gradient_norm_ratio_max"],
        weight_relative_error_max=hb["verification"]["weight_relative_error_max"],
        min_params_changed_ratio=hb["verification"]["min_params_changed_ratio"],
        # MFU settings
        gpu_peak_tflops=hb["mfu"]["gpu_peak_tflops"],
        min_mfu=hb["mfu"]["min_mfu"],
        max_plausible_mfu=hb["mfu"]["max_plausible_mfu"],
        # Timer integrity check
        timer_divergence_threshold=hb["verification"]["timer_divergence_threshold"],
    )
    
    # 5. Output Result
    print("\n" + "="*50)
    print("VALIDATOR RESULT")
    print("="*50)
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Error: {result.get('error')}")
        print(f"Error Code: {result.get('error_code')}")
    
    print(f"MFU: {result.get('mfu', 0.0):.2f}%")
    print(f"TPS: {result.get('tps', 0.0):.2f}")
    
    print("\nDiagnostics:")
    # Clean up non-serializable objects from diagnostics (tensors, etc.)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif hasattr(obj, 'tolist'): # numpy and torch tensors
            return obj.tolist()
        elif hasattr(obj, 'detach'): # torch tensors (if tolist not present)
            return obj.cpu().detach().tolist()
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj

    diag = sanitize(result.get('diagnostics', {}))
    print(json.dumps(diag, indent=2))
    print("="*50)

if __name__ == "__main__":
    asyncio.run(simulate())
