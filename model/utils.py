import pathlib
import torch
import numpy as np
import importlib_metadata
import pandas as pd
import os
from torchvision.ops import box_convert
from typing import List, Tuple, Dict
from neptunecontrib.api import log_table
from metrics.bounding_box import BoundingBox
from metrics.enumerators import BBFormat, BBType


def get_filenames_of_path(path:pathlib.Path, ext:str = "*") -> List[pathlib.Path]:
    """Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

def from_dict_to_boundingbox(file: dict, name: str, groundtruth: bool = True):
    """Returns list of BoundingBox objects from groundtruth or prediction.
    """
    labels = file["labels"]
    boxes = file["boxes"]
    scores = np.array(file["scores"].cpu()) if not groundtruth else [None] * len(boxes)
    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED
    return [
        BoundingBox(
            image_name=name,
            class_id=int(l),
            coordinates=tuple(bb),
            format=BBFormat.XYX2Y2,
            bb_type=gt,
            confidence=s,
        )
        for bb, l, s in zip(boxes, labels, scores)
    ]

#=========== DATA SET AUXILIARY FUNCTIONS ===========

def process_image(image:np.ndarray) -> torch.Tensor:
    """Return a torch tensor image in RGB loaded from a path.
    """
    if len(image.shape) == 4: # convert RGBA -> RGB
        image = image.convert('RGB')
    return image

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
            ).type(torch.float64)
    if box_output_format:
        boxes = box_convert(boxes, in_fmt=box_input_format, out_fmt=box_output_format).numpy()
    else:
        boxes = boxes.numpy()
    labels = np.array(
                    [label_indexes[object["classTitle"]] for object in target["objects"]],
                    dtype=np.int64
                 )
    return (boxes, labels)

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


#=========== DATA LOADER FUNCTIONS ===========

def collate_double(batch:Dict[torch.tensor, str]) -> Tuple[torch.tensor, str]:
    """
    collate function for the ObjectsDataSet.
    Only used by the dataloader.
    """
    images = [sample['img'].type(torch.double) for sample in batch]
    boxes_labelled = [sample['boxes_labelled'] for sample in batch]
    images_name = [sample['img_file_name'] for sample in batch]
    target_file_name = [sample['target_file_name'] for sample in batch]
    return images, boxes_labelled, images_name, target_file_name

#=========== MODEL LOGGER ===========

def log_packages_neptune(neptune_logger):
    """Uses the neptunecontrib.api to log the packages of the current python env."""
    dists = importlib_metadata.distributions()
    packages = {
        idx: (dist.metadata["Name"], dist.version) for idx, dist in enumerate(dists)
    }
    packages_df = pd.DataFrame.from_dict(
        packages, orient="index", columns=["package", "version"]
    )
    log_table(name="packages", table=packages_df, experiment=neptune_logger.experiment)

def log_mapping_neptune(mapping: dict, neptune_logger):
    """Uses the neptunecontrib.api to log a class mapping."""
    mapping_df = pd.DataFrame.from_dict(
        mapping, orient="index", columns=["class_value"]
    )
    log_table(name="mapping", table=mapping_df, experiment=neptune_logger.experiment)

def log_model_neptune(
    checkpoint_path: pathlib.Path,
    save_directory: pathlib.Path,
    name: str,
    neptune_logger,
):
    """Saves the model to disk, uploads it to neptune and removes it again."""
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["hyper_parameters"]["model"]
    torch.save(model.state_dict(), save_directory / name)
    neptune_logger.experiment.set_property("checkpoint_name", checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(save_directory / name))
    if os.path.isfile(save_directory / name):
        os.remove(save_directory / name)

def log_checkpoint_neptune(checkpoint_path: pathlib.Path, neptune_logger):
    neptune_logger.experiment.set_property("checkpoint_name", checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(checkpoint_path))