"""The gate-toy command of v0.1.0: the same as python -m gate_distill.demo (gate-demo)."""
from .demo import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
