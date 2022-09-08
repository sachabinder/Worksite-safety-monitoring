import pathlib
import albumentations as A
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
)
from model.utils import (
    collate_double,
    get_filenames_of_path,
    log_mapping_neptune,
    log_model_neptune,
    log_packages_neptune,
)
from model.dataset import ObjectsDataSet
from model.backbone_resnet import ResNetBackbones
from model.transformation import (
    ComposeDouble,
    Clip,
    AlbumentationWrapper,
    FunctionWrapperDouble,
)
from model.utils import (
    get_filenames_of_path, 
    collate_double,
    normalize_01
)
from model.faster_rcnn import get_faster_rcnn_resnet,FasterRCNNlightning

#=========== GLOBAL PARAMS ===========
USE_CACHE = False
LABEL_INDEXES = {
    "__background__":0,
    "People":1,
    "Vertical_formwork":2,
    "Rebars":3,
    "Shoring":4,
    "Concrete_pump_hose":5,
    "Mixer_truck":6
}

@dataclass
class Paths:
    IMG_TRAINING = pathlib.Path("Detection_Train_Set/Detection_Train_Set_Img")
    TARGETS_TRAINING = pathlib.Path("Detection_Train_Set/Detection_Train_Set_Json")
    IMG_TEST = pathlib.Path("Detection_Test_Set/Detection_Test_Set_Img")
    TARGETS_TEST = pathlib.Path("Detection_Test_Set/Detection_Test_Set_Json")


@dataclass
class Params:
    BATCH_SIZE: int = 1
    OWNER: str = "sachabinder"  # set your name here
    PROJECT: str = "worksite-safety-monitoring" # project name
    SAVE_DIR: Optional[str] = "../experiments"  # checkpoints will be saved to cwd (current working directory)
    LOG_MODEL: bool = False  # whether to log the model to neptune after training
    GPU: Optional[int] = None  # set to None for cpu training
    LR: float = 0.001 # learning rage
    PRECISION: int = 32
    CLASSES: int = 7 # number of classes
    SEED: int = 42
    MAXEPOCHS: int = 500 # max number of epochs
    PATIENCE: int = 50
    BACKBONE: ResNetBackbones = ResNetBackbones.RESNET50 # model backbone
    FPN: bool = False # FPN at the top
    ANCHOR_SIZE: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),)
    ASPECT_RATIOS: Tuple[Tuple[float, ...]] = ((0.5, 1.0, 2.0),)
    MIN_SIZE: int = 500 # min size
    MAX_SIZE: int = 2025 # max size
    IMG_MEAN: List = field(default_factory=lambda: [0.485, 0.456, 0.406]) # mean img param
    IMG_STD: List = field(default_factory=lambda: [0.229, 0.224, 0.225]) # std img param
    IOU_THRESHOLD: float = 0.5 # threshold intersection over union fox boxes

def main():
    params = Params()
    #=========== DATASET SETUP ===========
    # input and target files path
    inputs_train = get_filenames_of_path(Paths.IMG_TRAINING)
    targets_train = get_filenames_of_path(Paths.TARGETS_TRAINING)
    input_test_and_valid = get_filenames_of_path(Paths.IMG_TEST)
    targets_test_and_valid = get_filenames_of_path(Paths.TARGETS_TEST)
    inputs_train.sort()
    targets_train.sort()
    input_test_and_valid.sort()
    targets_test_and_valid.sort()
    # random seed
    seed_everything(params.SEED)
    # validation and test split
    inputs_valid = input_test_and_valid[:len(input_test_and_valid)//2]
    inputs_test = input_test_and_valid[len(input_test_and_valid)//2:]
    targets_valid = targets_test_and_valid[:len(input_test_and_valid)//2]
    targets_test = targets_test_and_valid[len(input_test_and_valid)//2:]
    #=========== TRANSFORMATION AND AUGMENTATION ===========
    # training
    transforms_training = ComposeDouble([
        Clip(),
        #AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
        #AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
        # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])
    # validation
    transforms_validation = ComposeDouble([
        Clip(),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])
    # test transformations
    transforms_test = ComposeDouble([
        Clip(),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])
    #=========== DATASET DECLARATION ===========
    # dataset training
    dataset_train = ObjectsDataSet(image_paths=inputs_train,
                                target_paths=targets_train,
                                label_indexes=LABEL_INDEXES,
                                transform=transforms_training,
                                use_cache=USE_CACHE)
    # dataset validation
    dataset_valid = ObjectsDataSet(image_paths=inputs_valid,
                                target_paths=targets_valid,
                                label_indexes=LABEL_INDEXES,
                                transform=transforms_validation,
                                use_cache=USE_CACHE)
    # dataset test
    dataset_test = ObjectsDataSet(image_paths=inputs_test,
                                target_paths=targets_test,
                                label_indexes=LABEL_INDEXES,
                                transform=transforms_test,
                                use_cache=USE_CACHE)
    #=========== DATASET LOADING ===========
    # dataloader training
    dataloader_train = DataLoader(dataset=dataset_train,
                                batch_size=params.BATCH_SIZE,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=collate_double)
    # dataloader validation
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate_double)
    # dataloader test
    dataloader_test = DataLoader(dataset=dataset_test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate_double)
    # neptune logger
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZTBkYWFjZC04ZjFjLTQ2NWQtOTBkZi0yMmUxYjMwYTI4NzcifQ==",
        project_name=f"{params.OWNER}/{params.PROJECT}",  # use your neptune name here
        experiment_name=params.PROJECT,
        params=params.__dict__,
    )
    assert neptune_logger.name  # http GET request to check if the project exists
    # model init
    model = get_faster_rcnn_resnet(
        num_classes=params.CLASSES,
        backbone_name=params.BACKBONE,
        anchor_size=params.ANCHOR_SIZE,
        aspect_ratios=params.ASPECT_RATIOS,
        fpn=params.FPN,
        min_size=params.MIN_SIZE,
        max_size=params.MAX_SIZE,
    )
    # lightning init
    task = FasterRCNNlightning(
        model=model, lr=params.LR, iou_threshold=params.IOU_THRESHOLD
    )
    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor="Validation_mAP", mode="max")
    learningrate_callback = LearningRateLogger()
    early_stopping_callback = EarlyStopping(
        monitor="Validation_mAP", patience=params.PATIENCE, mode="max"
    )
    # trainer init
    trainer = Trainer(
        gpus=params.GPU,
        precision=params.PRECISION,  # try 16 with enable_pl_optimizer=False
        callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        default_root_dir=params.SAVE_DIR,  # where checkpoints are saved to
        logger=neptune_logger,
        num_sanity_val_steps=0,
        max_epochs=params.MAXEPOCHS,
        log_save_interval=1
    )
    # start training
    trainer.fit(
        model=task, train_dataloader=dataloader_train, val_dataloaders=dataloader_valid
    )
    # start testing
    trainer.test(ckpt_path="best", dataloaders=dataloader_test)
    # log packages
    log_packages_neptune(neptune_logger=neptune_logger)
    # log mapping as table
    log_mapping_neptune(mapping=LABEL_INDEXES, neptune_logger=neptune_logger)
    # log model
    if params.LOG_MODEL:
        checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
        log_model_neptune(
            checkpoint_path=checkpoint_path,
            save_directory=pathlib.Path.home(),
            name="best_model.pt",
            neptune_logger=neptune_logger,
        )
    # stop logger
    neptune_logger.experiment.stop()
    print("Finished")

if __name__ == "__main__":
    main()