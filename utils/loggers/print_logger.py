from pprint import pprint
from typing import Any, Dict, List

from .base_logger import BaseLogger


class PrintLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super(PrintLogger, self).__init__(*args, **kwargs)
        import numpy as np
        from matplotlib import pyplot as plt
        from PIL.Image import Image

        self.Image = Image
        self.plt = plt
        self.np = np

    def stop(self):
        pass

    def log(self, name: str, data: Any, step=None):
        print(f"{name}: {data}" if step is None else f"step {step}, {name}: {data:.4e}")

    def _log_fig(self, name: str, fig: Any):
        if isinstance(fig, self.Image):
            fig = self.np.asarray(fig)
            self.plt.imshow(fig)
        elif isinstance(fig, self.np.ndarray):
            self.plt.imshow(fig)
        else:
            self.plt.show()

    def log_hparams(self, params: Dict[str, Any]):
        print("hyperparameters:")
        pprint(params)

    def log_params(self, params: Dict[str, Any]):
        print("params:")
        pprint(params)

    def add_tags(self, tags: List[str]):
        print("tags:")
        pprint(tags)

    def log_name_params(self, name: str, params: Any):
        print(f"{name}:")
        pprint(params)


class LoggerL(PrintLogger):
    def __init__(self, stdout, format=None, *args, **kwargs):
        super(LoggerL, self).__init__(*args, **kwargs)
        import logging

        from matplotlib import pyplot as plt

        self.show = plt.show
        handler = logging.StreamHandler(stdout)
        if format is None:
            format = "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        handler.setFormatter(logging.Formatter(format))
        self.logger = logging.getLogger()
        self.logger.addHandler(handler)
        self.logger.setLevel("INFO")
        self.logging = logging

    def log(self, text: str, data: Any, step=None):
        self.logging.info(text % data)
