from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv2

from random import randint


class Transform(ABC):
    """
    Basic Transform interface
    Applies desired transform to the given image
    """

    @abstractmethod
    def transform(self, image: np.ndarray):
        pass


class StrikeThrough(Transform):
    """
    Strike-through transform
    Rotates the given image, reduces opacity
    and merges with the initial one. Creates an effect
    of ink printed on the opposite side of the paper

    :param intensity - controls opacity of the background ink
    """
    def __init__(self, intensity=0.1):
        self.intensity = intensity

    def transform(self, image: np.ndarray):
        image = 1 - image

        back_image = cv2.rotate(image, cv2.ROTATE_180)
        back_image = np.clip(back_image * self.intensity, 0, 1)
        image = back_image + image

        image = 1 - image

        return image


class ContrastChange(Transform):
    """
    Contrast change transform
    Makes ink look more pale and old

    :param alpha - changes intensity of mids
    :param beta  - changes lower level for shadows
    """
    def __init__(self, alpha: float = 1, beta: float = 0.2):
        self.alpha = alpha
        self.beta = beta

    def transform(self, image: np.ndarray):
        image = np.clip(self.alpha * image + self.beta, 0, 1)

        return image


class Bleed(Transform):
    """
    Bleed transform
    Makes ink look printed and bleed

    :param strength - bigger the parameter more bold the text will be
    :param threshold - lower the parameter less ink the printer has
    """
    def __init__(self, strength: int = 5, threshold: float = 0.7):
        self._strength = strength
        self._threshold = threshold

    def transform(self, image: np.ndarray):
        gaussian_kernel_size = (self._strength, self._strength)
        image = cv2.GaussianBlur(image, gaussian_kernel_size, cv2.BORDER_DEFAULT)
        channels = list(cv2.split(image))
        for i, channel in enumerate(channels):
            _, threshold = cv2.threshold(channel, self._threshold, 1, cv2.THRESH_BINARY)
            channels[i] = np.clip(threshold, a_min=0, a_max=1)
        image = cv2.merge(channels)

        return image


class WhiteLines(Transform):
    """
    White lines transform
    Generates horizontal white lines as if the paper was printed on an old printer
    that has little ink

    :param intensity - intensity of the lines
    :param max_width - maximum width of the lines
    :param number_of_lines - number of lines
    """
    def __init__(self, intensity=1, max_width=8, number_of_lines=25):
        self.intensity = intensity
        self.max_width = max_width
        self.number_of_lines = number_of_lines

    def transform(self, image: np.ndarray):
        number_of_lines = randint(1, self.number_of_lines)
        lines = np.random.uniform(0, image.shape[0], number_of_lines)

        layer = np.zeros_like(image)
        for line_y in lines:
            line_y = int(line_y)
            width = randint(2, min(self.max_width, image.shape[0] - line_y))
            layer[line_y: line_y + width, :, :] += self.intensity * np.ones((width, layer.shape[1], layer.shape[2]))

        image += layer
        image = np.clip(image, 0, 1)
        return image


class InkDrops(Transform):
    """
    Effect of ink drops.

    :param max_x  - maximum x for random generator
    :param max_y  - maximum y for random generator
    :param number - number of drops to generate
    """
    def __init__(self, max_x: int, max_y: int, number=1):
        self.locs = [
            (randint(0, max_y), randint(0, max_x))
            for _ in range(number)
        ]

    def transform(self, image: np.ndarray):
        layer = np.zeros_like(image)
        for loc in self.locs:
            cov = [[image.shape[0], 0], [0, image.shape[1]]]
            mean = loc

            dots = np.random.multivariate_normal(mean, cov, 1000).astype(int)

            for y, x in dots:
                if layer.shape[0] > y > 0 and layer.shape[1] > x > 0:
                    layer[y, x, :] = [1, 1, 1]

        layer = Bleed(strength=51, threshold=0.1).transform(layer)
        layer = cv2.GaussianBlur(layer, (3, 3), cv2.BORDER_DEFAULT)

        layer = 1 - layer
        layer = ContrastChange(alpha=0.9, beta=0).transform(layer)
        layer = 1 - layer

        image = 1 - image
        image = image + layer
        image = 1 - image
        image = np.clip(image, 0, 1)
        return image


class Lightning(Transform):
    """
    Effect of lamp lightning.

    :param max_x  - maximum x for random generator
    :param max_y  - maximum y for random generator
    :param number - number of drops to generate
    """
    def __init__(self, max_x: int, max_y: int, number=1):
        self.locs = [
            (randint(0, max_y), randint(0, max_x))
            for _ in range(number)
        ]

    def transform(self, image: np.ndarray):
        layer = np.zeros_like(image)
        for loc in self.locs:
            cov = [[image.shape[0] * 5, 0], [0, image.shape[1] * 5]]
            mean = loc

            dots = np.random.multivariate_normal(mean, cov, 10).astype(int)

            for y, x in dots:
                if layer.shape[0] > y > 0 and layer.shape[1] > x > 0:
                    layer = cv2.circle(layer, (y, x), 700, (1, 1, 1), -1)

        kernel = np.ones((35, 35), np.uint8)
        layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)
        layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)

        layer = Bleed(strength=101, threshold=0).transform(layer)

        for i in range(10):
            layer = cv2.GaussianBlur(layer, (11, 11), cv2.BORDER_DEFAULT)

        layer = ContrastChange(beta=0, alpha=0.1).transform(layer)
        image = image + layer
        image = np.clip(image, 0, 1)
        return image


class CameraFocus(Transform):
    """
    Effect of camera focus loss
    """
    def __init__(self, strength: int=3):
        self.strength = strength

    def transform(self, image: np.ndarray):
        image = cv2.GaussianBlur(image, (self.strength, self.strength), cv2.BORDER_DEFAULT)
        return image
