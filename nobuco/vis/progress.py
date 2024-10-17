from tqdm import tqdm
from rich.console import Console
import time
import warnings

class ProgressBar:
    def __init__(self, prefix, total=None, bar_format=None):
        self.prefix = prefix
        self.total = total
        self.console = Console()
        self.start_time = time.time()
        
        try:
            # Attempt to use tqdm.rich
            from tqdm.rich import tqdm as rich_tqdm
            self.bar = rich_tqdm(
                total=total,
                desc=f"[Nobuco] {self.prefix}",
                bar_format=bar_format,
                colour="blue",
                unit="ops"
            )
        except Exception:
            # Fallback to standard tqdm
            warnings.warn("Failed to initialize tqdm.rich. Falling back to standard tqdm.", RuntimeWarning)
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
            self.bar.set_description_str(f"[Nobuco] {self.prefix} (DONE)")
            self.bar.close()
        except Exception:
            warnings.warn("Error occurred while closing the progress bar.", RuntimeWarning)
        
        elapsed = time.time() - self.start_time
        self.console.print(f"[bold green][Nobuco] {self.prefix} (DONE) - Elapsed time: {elapsed:.2f} sec")

    def __del__(self):
        try:
            self.close()
        except:
            pass  # Ignore any errors during garbage collection
