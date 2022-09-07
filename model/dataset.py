import torch
import json
import pathlib
import cv2
from torchvision.ops import box_convert
from typing import List, Dict
from utils import (get_filenames_of_path, 
                   process_image,
                   get_boxes_and_labels_from_target)


IMG_FOLDER = pathlib.Path("Detection_Train_Set/Detection_Train_Set_Img")
TARGETS_FOLDER = pathlib.Path("Detection_Train_Set/Detection_Train_Set_Json")
LABEL_INDEXES = {
    "__background__":0,
    "People":1,
    "Vertical_formwork":2,
    "Rebars":3,
    "Shoring":4,
    "Concrete_pump_hose":5
}


class ObectsDataSet(torch.utils.data.Dataset):
    """Build a dataset of boxes with their corresponding labels with
    image and its corresponding target which should be a JSON file with
    the same name of the image.
    """
    def __init__(self,
                 image_paths: List[pathlib.Path],
                 target_paths: List[pathlib.Path],
                 label_indexes: Dict,
                 box_convert_to_format: str = "xywh") -> None:
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.box_format_to_convert = box_convert_to_format
        self.label_indexes = label_indexes
        self.box_format = "xyxy"

    def __len__(self):
        """Definition of operator len() for the dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, index):
        orig_image, target = self.extract_img_and_target_from_path(img_path=self.image_paths[index],
                                                                   target_path=self.target_paths[index])
        image = process_image(image=orig_image)
        boxes, labels = get_boxes_and_labels_from_target(target=target,
                                                         box_input_format=self.box_format,
                                                         box_output_format=self.box_format_to_convert,
                                                         label_indexes=self.label_indexes)
        return {'img':image,
                'img_file_name':self.image_paths[index].name,
                'target':{'boxes': boxes,'labels': labels},
                'target_file_name':self.target_paths[index].name}
    
    @staticmethod
    def extract_img_and_target_from_path(img_path, target_path):
        with open(target_path, "r") as f:
            target = json.load(f)
        return cv2.imread(str(img_path)), target


if __name__ == "__main__":
    targets = get_filenames_of_path(TARGETS_FOLDER)
    images = get_filenames_of_path(IMG_FOLDER)
    targets.sort()
    images.sort()

    data_set = ObectsDataSet(images, targets,LABEL_INDEXES)
    print(data_set[0])
