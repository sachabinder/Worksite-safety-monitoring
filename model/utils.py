import pathlib
import torch
import numpy as np
from torchvision.ops import box_convert
from typing import List, Tuple, Dict


def get_filenames_of_path(path:pathlib.Path, ext:str = "*") -> List[pathlib.Path]:
    """Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames


#=========== DATA SET AUXILIARY FUNCTIONS ===========

def process_image(image:np.ndarray) -> torch.Tensor:
    """Return a torch tensor image in RGB loaded from a path.
    """
    if len(image.shape) == 4: # convert RGBA -> RGB
        image = image.convert('RGB')
    return torch.from_numpy(image).type(torch.float32)

def get_boxes_and_labels_from_target(target:Dict,
                                     box_input_format:str,
                                     box_output_format:str,
                                     label_indexes:Dict) -> Tuple[np.ndarray]:
    """Return a np array couple of (boxes, labels), 
    convert format : box_input_format -> box_output_format,
    from a polygon loaded through JSON file.
    """
    boxes = torch.tensor(
                [get_bounding_box(object["points"]["exterior"]) for object in target["objects"]]
            ).type(torch.float32)
    if boxes.numel() != 0:
        boxes_converted = box_convert(boxes, in_fmt=box_input_format, out_fmt=box_output_format)
        boxes_converted = boxes_converted.numpy()
    else:
        boxes_converted = boxes.numpy()
    labels = np.array(
                    [label_indexes[object["classTitle"]] for object in target["objects"]],
                    dtype=np.int64
                 )
    return (boxes_converted, labels)

def get_bounding_box(poligon_points:List[List[int]]) -> List[int]:
    """Compute coordinates of bounding box of a polygon.
    Return points with the format xyxy.
    (x_1,y_1)-------------------|
        |                       |
        |       Polygon         |
        |                       |
        ---------------------(x_2,y_2)
    """
    x_coords = [point[0] for point in poligon_points]
    y_coords = [point[1] for point in poligon_points] 
    x_1, y_1, x_2, y_2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    return [x_1,y_1,x_2,y_2]


#=========== TRANSFORMATION FUNCTIONS ===========

def clip_bounding_boxes(input:np.ndarray, target:np.ndarray) -> np.array:
    """
    If the bounding boxes exceed one dimension, they are clipped to the dim's maximum.
    Bounding boxes are expected to be in xyxy format.
    Example: x_value=224 but x_shape=200 -> x1=199
    """
    output = []
    for bounding_boxe in target:
        x_1, y_1, x_2, y_2 = tuple(bounding_boxe)
        x_shape = input.shape[1]
        y_shape = input.shape[0]
        x_1, y_1, x_2, y_2 = clip(x_1, x_shape), clip(y_1, y_shape), clip(x_2, x_shape), clip(y_2, y_shape)
        output.append([x_1, y_1, x_2, y_2])
    return np.array(output)

def clip(value: int, max: int):
    if value >= max - 1:
        value = max - 1
    elif value <= 0:
        value = 0
    return value

def normalize_01(input: np.ndarray) -> np.ndarray:
    """Squash image input to the value range [0, 1] (no clipping)"""
    input_out = (input - np.min(input)) / np.ptp(input)
    return input_out