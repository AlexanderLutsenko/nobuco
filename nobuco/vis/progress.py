import sys
from tqdm import tqdm


class ProgressBar:
    def __init__(self, prefix, total=None, bar_format=None):
        self.prefix = prefix
        self.bar = tqdm(file=sys.stdout, total=total, bar_format=bar_format)
        self.bar.set_description_str(f"[Nobuco] {self.prefix}")

    def update(self, n=1):
        self.bar.update(n)

    def close(self):
        self.bar.set_description_str(f"[Nobuco] {self.prefix} (DONE)")
        self.bar.close()
