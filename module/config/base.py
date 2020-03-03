from pathlib import Path
Base_Dir = Path("module")
config = {
    "data_dir": Base_Dir / "dataset",
    "log_dir": Base_Dir / "output/log",
    "write_dir": Base_Dir / "output/TSboard",
    "figure_dir": Base_Dir / "output/figure",
    "checkpoint_dir": Base_Dir / "output/checkpoints",
    "cache_dir": Base_Dir / 'model/',
    "result_dir": Base_Dir / "output/result"
}

