"""Code adapted from
https://gist.githubusercontent.com/johschmidt42/a0b7d9380e7c8278decb63defb1e9eb2/raw/bbef128239a01ba8500c831d1b2829de1322b0c2/Faster_RCNN_Model.py
@author : @johschmidt42 
"""
import torch
from itertools import chain
import pytorch_lightning as pl
from model.utils import from_dict_to_BoundingBox


class FasterRCNN_lightning(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 0.0001,
                 iou_threshold: float = 0.5 # iou = intersection over union
                 ):
        super().__init__()
        # Model
        self.model = model
        # Classes (background inclusive)
        self.num_classes = self.model.num_classes
        # Learning rate
        self.lr = lr
        # IoU threshold
        self.iou_threshold = iou_threshold
        # Transformation parameters
        self.mean = model.image_mean
        self.std = model.image_std
        self.min_size = model.min_size
        self.max_size = model.max_size
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, image):
        self.model.eval()
        return self.model(image)

    def training_step(self, batch, batch_idx):
        # Batch
        image, boxes_labelled, image_name, label_name = batch  # tuple unpacking
        loss_dict = self.model(image, boxes_labelled)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch
        image, boxes_labelled, image_name, label_name = batch  # tuple unpacking
        # Inference
        preds = self.model(image)
        exact_boxes = [from_dict_to_BoundingBox(target, name=name, groundtruth=True) for target, name in zip(boxes_labelled, image_name)]
        exact_boxes = list(chain(*exact_boxes))
        pred_boxes = [from_dict_to_BoundingBox(pred, name=name, groundtruth=False) for pred, name in zip(preds, image_name)]
        pred_boxes = list(chain(*pred_boxes))
        return {'pred_boxes': pred_boxes, 'exact_boxes': exact_boxes}

    def validation_epoch_end(self, outs):
        exact_boxes = [out['exact_boxes'] for out in outs]
        exact_boxes = list(chain(*exact_boxes))
        pred_boxes = [out['pred_boxes'] for out in outs]
        pred_boxes = list(chain(*pred_boxes))
        from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
        from metrics.enumerators import MethodAveragePrecision
        metric = get_pascalvoc_metrics(gt_boxes=exact_boxes,
                                       det_boxes=pred_boxes,
                                       iou_threshold=self.iou_threshold,
                                       method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table=True)
        per_class, mAP = metric['per_class'], metric['mAP']
        self.log('Validation_mAP', mAP)
        for key, value in per_class.items():
            self.log(f'Validation_AP_{key}', value['AP'])

    def test_step(self, batch, batch_idx):
        # Batch
        image, boxes_labelled, image_name, label_name = batch  # tuple unpacking
        # Inference
        preds = self.model(image)
        exact_boxes = [from_dict_to_BoundingBox(target, name=name, groundtruth=True) for target, name in zip(boxes_labelled, image_name)]
        exact_boxes = list(chain(*exact_boxes))
        pred_boxes = [from_dict_to_BoundingBox(pred, name=name, groundtruth=False) for pred, name in zip(preds, image_name)]
        pred_boxes = list(chain(*pred_boxes))
        return {'pred_boxes': pred_boxes, 'exact_boxes': exact_boxes}

    def test_epoch_end(self, outs): #TODO check exact_boxes
        exact_boxes = [out['exact_boxes'] for out in outs]
        exact_boxes = list(chain(*exact_boxes))
        pred_boxes = [out['pred_boxes'] for out in outs]
        pred_boxes = list(chain(*pred_boxes))
        from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
        from metrics.enumerators import MethodAveragePrecision
        metric = get_pascalvoc_metrics(gt_boxes=exact_boxes,
                                       det_boxes=pred_boxes,
                                       iou_threshold=self.iou_threshold,
                                       method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table=True)
        per_class, mAP = metric['per_class'], metric['mAP']
        self.log('Test_mAP', mAP)
        for key, value in per_class.items():
            self.log(f'Test_AP_{key}', value['AP'])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=0.005)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  factor=0.75,
                                                                  patience=30,
                                                                  min_lr=0)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'Validation_mAP'}