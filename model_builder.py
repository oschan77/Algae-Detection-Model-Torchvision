import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# This function returns a Faster R-CNN model pre-trained on COCO dataset
# with the final fully-connected layer replaced to fit the number of classes
# of the user's specific application.
# If feature_extract is True, only the newly added layer is trainable.
# Otherwise, the whole model is trainable.
def get_faster_rcnn_model(num_classes, feature_extract=True):
  # Load the pre-trained Faster R-CNN model
  weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
  
  # If feature_extract is True, freeze all model parameters
  if feature_extract:
      for param in model.parameters():
          param.requires_grad = False

  # Get the number of input features of the final fully-connected layer
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # Replace the final fully-connected layer with a new one that fits the
  # number of classes of the user's specific application
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model
