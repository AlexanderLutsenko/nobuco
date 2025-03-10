from tqdm import tqdm
import time
import warnings


class ProgressBar:
    def __init__(self, prefix, total=None, bar_format=None):
        self.prefix = prefix
        self.total = total
        self.bar = tqdm(
            total=total,
            desc=f"[Nobuco] {self.prefix}",
            bar_format=bar_format,
            unit="ops"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self, n=1):
        try:
            self.bar.update(n)
        except Exception:
            pass  # Silently ignore update errors

    def close(self):
        try:
            self.bar.set_description(f"[Nobuco] {self.prefix} (DONE)")
            self.bar.close()
        except Exception:
            warnings.warn("Error occurred while closing the progress bar.", RuntimeWarning)
        
    def __del__(self):
        try:
            self.close()
        except:
            pass  # Ignore any errors during garbage collection
