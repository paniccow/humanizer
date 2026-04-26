"""Run the main GRPO training. See humanizer.train.grpo.GRPOConfig for knobs."""
from humanizer.train.grpo import GRPOConfig, run

if __name__ == "__main__":
    out = run(GRPOConfig())
    print(f"GRPO adapter saved to: {out}")
