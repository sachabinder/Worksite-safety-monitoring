import torch
import json
import pathlib
import cv2
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_area
from model.utils import (process_image,
                   get_boxes_and_labels_from_target,
                   get_filenames_of_path,
                   normalize_01)
from model.transformation import (ComposeDouble,
                            Clip,
                            FunctionWrapperDouble)


IMG_FOLDER = pathlib.Path("Detection_Train_Set/Detection_Train_Set_Img")
TARGETS_FOLDER = pathlib.Path("Detection_Train_Set/Detection_Train_Set_Json")
LABEL_INDEXES = {
    "__background__":0,
    "People":1,
    "Vertical_formwork":2,
    "Rebars":3,
    "Shoring":4,
    "Concrete_pump_hose":5,
    "Mixer_truck":6
}


class ObjectsDataSet(torch.utils.data.Dataset):
    """Build a dataset of boxes with their corresponding labels with
    image and its corresponding target which should be a JSON file with
    the same name of the image.
    """

    def __init__(self,
                 image_paths:List[pathlib.Path],
                 target_paths:List[pathlib.Path],
                 label_indexes:Dict,
                 transform = None,
                 use_cache:bool=False,
                 box_convert_to_format:str = None) -> None:
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.box_format_to_convert = box_convert_to_format
        self.label_indexes = label_indexes
        self.transform = transform
        self.use_cache = use_cache
        if self.use_cache:  # load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.extract_img_and_target_from_path,
                                                zip(image_paths, target_paths))
        self.box_format = "xyxy"

    def __len__(self):
        """Definition of operator len() for the dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, index:int) -> Dict[torch.tensor, str]:
        if self.use_cache:
            orig_image, target = self.cached_data[index]
        else:
            orig_image, target = self.extract_img_and_target_from_path(img_path=self.image_paths[index],
                                                                       target_path=self.target_paths[index])
        image = process_image(image=orig_image)
        boxes, labels = get_boxes_and_labels_from_target(target=target,
                                                         box_input_format=self.box_format,
                                                         box_output_format=self.box_format_to_convert,
                                                         label_indexes=self.label_indexes)
        boxes_labelled = {'boxes': boxes, 'labels': labels}
        if self.transform:  # apply transformations
            image_transformed, boxes_labelled_transformed = self.transform(image, boxes_labelled)
        # Convert to torch tensor
        image_transformed = torch.from_numpy(image_transformed)
        boxes_labelled_transformed = {key: torch.from_numpy(value) 
                                        for key, value in boxes_labelled_transformed.items()}
        return {'img':image_transformed,
                'img_file_name':self.image_paths[index].name,
                'boxes_labelled': boxes_labelled_transformed,
                'target_file_name':self.target_paths[index].name}
    
    @staticmethod
    def extract_img_and_target_from_path(img_path:pathlib.Path,
                                         target_path:pathlib.Path) -> Tuple:
        with open(target_path, "r") as f:
            target = json.load(f)
        return cv2.imread(str(img_path)), target


def dataset_builder() -> torch.utils.data.Dataset:
    images_fles = get_filenames_of_path(IMG_FOLDER)
    targets_files = get_filenames_of_path(TARGETS_FOLDER)
    images_fles.sort()
    targets_files.sort()
    # adaptation and augmentation of dataset
    transforms = ComposeDouble([Clip(),
                                FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
                                FunctionWrapperDouble(normalize_01)])
    dataset = ObjectsDataSet(image_paths=images_fles,
                             target_paths=targets_files,
                             label_indexes=LABEL_INDEXES,
                             transform=transforms)
    return dataset

def dataset_stats(dataset: torch.utils.data.Dataset,
                  rcnn_transform: GeneralizedRCNNTransform = False) -> Dict:
    """Iterates over the dataset and returns some stats.
    Can be useful to pick the right anchor box sizes.
    """
    stats = {
        "image_height": [],
        "image_width": [],
        "image_mean": [],
        "image_std": [],
        "boxes_height": [],
        "boxes_width": [],
        "boxes_num": [],
        "boxes_area": []
    }
    for batch in dataset:
        # Batch
        image, boxes_labelled = batch["img"], batch["boxes_labelled"]
        # Transform
        if rcnn_transform:
            image, boxes_labelled = rcnn_transform([image], [boxes_labelled])
            image, boxes_labelled = image.tensors, boxes_labelled[0]
        # Image
        stats["image_height"].append(image.shape[-2])
        stats["image_width"].append(image.shape[-1])
        stats["image_mean"].append(image.mean().item())
        stats["image_std"].append(image.std().item())
        # box
        wh = boxes_labelled["boxes"][:, -2:]
        stats["boxes_height"].append(wh[:, 1])
        stats["boxes_width"].append(wh[:, 0])
        stats["boxes_num"].append(len(wh))
        stats["boxes_area"].append(
            box_area(boxes_labelled["boxes"])
        )
    # convertion in torch tensor
    for key, val in stats.items():
        if key == "boxes_height" or key =="boxes_width":
            stats[key] = torch.cat(val)
    return stats