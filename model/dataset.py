from email.mime import image
from typing import List, Dict
import torch
from torchvision.ops import box_convert
import json
import pathlib
from utils import get_filenames_of_path
import cv2

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
                 label_index : Dict,
                 box_convert_to_format: str = "xywh") -> None:
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.box_convert_to_format = box_convert_to_format
        self.label_index = label_index
        self.box_format = "xyxy"

    def __len__(self):
        """Definition of operator len() for the dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # image process
        image = cv2.imread(str(self.image_paths[index]))
        if len(image.shape) == 4: # convert RGBA -> RGB
            image = image.convert('RGB')
        image = torch.from_numpy(image).type(torch.float32)
        # boxes extraction and process
        target = json.load(open(self.target_paths[index], "r"))
        boxes = torch.tensor(
                    [get_bounding_box(object["points"]["exterior"]) for object in target["objects"]]
                ).to(torch.float32)
        boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt=self.box_convert_to_format)
        # labels extraction
        labels = torch.tensor(
                    [LABEL_INDEXES[object["classTitle"]] for object in target["objects"]]
                 ).to(torch.int64)
        return {'img':image, 'img_file_name':self.image_paths[index].name, 'target':{'boxes': boxes,'labels': labels}, 'target_file_name':self.target_paths[index].name}


def get_bounding_box(poligon_points):
    """Compute coordinates of bounding box of a polygon.
    Return points with the format xyxy.
    (x_1,y_1)-------------------|
        |                       |
        |       Polygon         |
        |                       |
        ---------------------(x_2,y_2)
    """
    x_1 = min([point[0] for point in poligon_points])
    y_1 = min([point[1] for point in poligon_points])
    x_2 = max([point[0] for point in poligon_points])
    y_2 = max([point[1] for point in poligon_points])
    return [x_1,y_1,x_2,y_2]


if __name__ == "__main__":
    targets = get_filenames_of_path(TARGETS_FOLDER)
    images = get_filenames_of_path(IMG_FOLDER)
    targets.sort()
    images.sort()

    data_set = ObectsDataSet(images, targets)
    print(data_set[0])
