from augmentation.ink_transform import Transform
from random import random


class Pipeline:
    def __init__(self):
        self.pipeline = []

    def add(self, transform: Transform, probability: float = 1):
        if random() < probability:
            self.pipeline.append(transform)
        return self

    def apply(self, image):
        for transform in self.pipeline:
            image = transform.transform(image)
        return image
