"""Run SFT warm-start. See humanizer.train.sft.SFTConfig for knobs."""
from humanizer.train.sft import SFTConfig, run

if __name__ == "__main__":
    out = run(SFTConfig())
    print(f"SFT adapter saved to: {out}")
