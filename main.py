"""Unified command-line launcher for existing Phase 1-6 experiments."""
import argparse
import runpy
import sys

def main():
    parser = argparse.ArgumentParser(description="Unified Phase 1-6 research launcher")
    parser.add_argument("--phase", choices=["1", "2", "3", "4", "5", "6"], required=True)
    args, remaining = parser.parse_known_args()
    script = "run" if args.phase == "1" else ("phase6" if args.phase == "6" else "extended")
    extra = ["--phase", args.phase] if args.phase in ("2", "3", "4", "5") else []
    sys.argv = [script + ".py", *extra, *remaining]
    runpy.run_module(script, run_name="__main__")

if __name__ == "__main__":
    main()
