"""File adapted from : 
https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/blob/70886f36600fb0ccd5ef40f15644f932c9a72779/pytorch_faster_rcnn_tutorial/transformations.py
"""
import numpy as np
from typing import List, Callable, Tuple
from functools import partial
from model.utils import clip_bounding_boxes


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self):
        return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for image-target pairs."""

    def __call__(self, input:np.ndarray, target:dict) -> Tuple:
        for t in self.transforms:
            input, target = t(input, target)
        return input, target


class Repr:
    """Evaluatable string representation of an object"""

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


class Clip(Repr):
    """
    If the bounding boxes exceed one dimension, they are clipped to the dim's maximum.
    Bounding boxes are expected to be in xyxy format.
    Example: x_value=224 but x_shape=200 -> x1=199
    """

    def __call__(self, input:np.ndarray, target:dict):
        new_boxes = clip_bounding_boxes(input=input, target=target["boxes"])
        target["boxes"] = new_boxes
        return input, target


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(
        self,
        function: Callable,
        input: bool = True,
        target: bool = False,
        *args,
        **kwargs,
    ):
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, input: np.ndarray, target: dict):
        if self.input:
            input = self.function(input)
        if self.target:
            target = self.function(target)
        return input, target