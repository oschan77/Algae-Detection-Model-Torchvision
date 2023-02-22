import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from pathlib import Path

def apply_nms(orig_prediction, iou_thresh=0.3):
  keep = torchvision.ops.nms(orig_prediction['boxes'].cpu(), orig_prediction['scores'].cpu(), iou_thresh)
  
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'].cpu()[keep]
  final_prediction['scores'] = final_prediction['scores'].cpu()[keep]
  final_prediction['labels'] = final_prediction['labels'].cpu()[keep]
  
  return final_prediction


def torch_to_pil(img):
  return transforms.ToPILImage()(img).convert('RGB')


def plot_img_bbox(img, target, num_classes):
  fig, a = plt.subplots(1, 1)
  fig.set_size_inches(10, 10)
  a.imshow(img)

  for i in range(len(target['boxes'])):
    box = target['boxes'][i]
    label = int(target['labels'][i])

    cmap = plt.cm.get_cmap('hsv', num_classes+1)

    x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]

    rect = patches.Rectangle(
      (x, y),
      width, height,
      linewidth = 2,
      edgecolor = cmap(label),
      facecolor = 'none'
    )

    a.add_patch(rect)
  plt.show()


def inference_and_plot(
  dataset: torch.utils.data.Dataset,
  model: nn.Module,
  device: str,
  iou_thresh: float,
  num_classes: int,
):
  random_idx = random.randint(0, len(dataset)-1)
  img, target = dataset[random_idx]

  model.eval()
  with torch.inference_mode():
    prediction = model([img.to(device)])[0]

  nms_prediction = apply_nms(prediction, iou_thresh=iou_thresh)

  plot_img_bbox(torch_to_pil(img), nms_prediction, num_classes)


def save_model(
  model: torch.nn.Module,
  target_path: str,
  model_name: str,
):
  assert model_name.endswith('pth') or model_name.endswith('.pt'), "[Invalid model name]: model_name should end with '.pth' or '.pt'."

  target_path = Path(target_path)
  target_path.mkdir(parents=True, exist_ok=True)

  torch.save(
    obj = model.state_dict(),
    f = target_path / model_name,
  )
