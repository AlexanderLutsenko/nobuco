from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
import time

class ProgressBar:
    def __init__(self, prefix, total=None, bar_format=None):
        self.prefix = prefix
        self.total = total
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.elapsed]{task.elapsed}"),
            console=self.console,
            transient=True
        )
        self.task = None
        self.start_time = time.time()

    def __enter__(self):
        self.progress.start()
        self.task = self.progress.add_task(f"[bold blue][Nobuco] {self.prefix}", total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self, n=1):
        self.progress.update(self.task, advance=n)

    def close(self):
        elapsed = time.time() - self.start_time
        self.progress.stop()
        self.console.print(f"[bold green][Nobuco] {self.prefix} (DONE) - Elapsed time: {elapsed:.2f} sec")