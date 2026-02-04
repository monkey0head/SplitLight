import torch
from pytorch_lightning import seed_everything

def reset_peak_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()

def log_peak_memory_stats(task=None):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)      # GB
                max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)    # GB
                print(f"[GPU {i}] peak allocated: {max_alloc:.4f} GB | peak reserved: {max_reserved:.4f} GB")
                if task is not None:
                    task.get_logger().report_single_value(f'gpu{i}_peak_allocated_gb', max_alloc)
                    task.get_logger().report_single_value(f'gpu{i}_peak_reserved_gb', max_reserved)

def fix_seeds(random_state):
    """Set up random seeds."""

    seed_everything(random_state, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False