import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_FasterRCNN_model(num_classes, feature_extract=True):
  weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

  if feature_extract:
    for param in model.parameters():
      param.requires_grad = False

  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model
