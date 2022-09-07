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
    boxes = np.array(
                [get_bounding_box(object["points"]["exterior"]) for object in target["objects"]],
                dtype=np.float32
            )
    boxes_converted = box_convert(boxes, in_fmt=box_input_format, out_fmt=box_output_format)
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