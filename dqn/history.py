import numpy as np


class History:
    """
    This class represents a single state. It keeps a stack of video game frames.
    """

    def __init__(self, config):
        self.cnn_format = config.cnn_format

        batch_size, history_length = config.batch_size, config.history_length
        screen_height, screen_width = config.screen_height, config.screen_width

        self.history = np.zeros(
            [history_length, screen_height, screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            return np.transpose(self.history, (1, 2, 0))
        else:
            return self.history
