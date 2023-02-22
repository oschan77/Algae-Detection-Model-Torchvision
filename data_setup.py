import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms

class CocoDataset(CocoDetection):
  def __init__(self, root, annFile, transform=None, target_transform=None) -> None:
    super().__init__(root, annFile, transform, target_transform)

  def __getitem__(self, index: int):
    img, ori_target = super().__getitem__(index)

    num_objs = len(ori_target)
    boxes = []
    labels = []
    area = []
    iscrowd = []

    for i in range(num_objs):
      x_min = max(0, ori_target[i]['bbox'][0])
      y_min = max(0, ori_target[i]['bbox'][1])
      x_max = min(4908, x_min + ori_target[i]['bbox'][2])
      y_max = min(3264, y_min + ori_target[i]['bbox'][3])
      boxes.append([x_min, y_min, x_max, y_max])
      labels.append(ori_target[i]['category_id'])
      area.append(ori_target[i]['area'])
      iscrowd.append(ori_target[i]['iscrowd'])

    target = {}
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
    target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    target['image_id'] = torch.tensor([ori_target[0]['image_id']])
    target['area'] = torch.as_tensor(area, dtype=torch.float32)
    target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

    return img, target


def get_transform(train):
  if train:
    return transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(0.5),
    ])
  else:
    return transforms.Compose([
      transforms.ToTensor(),
    ])

def split_dataset(dataset_full, train_ratio):
  num_train = int(len(dataset_full) * train_ratio)
  num_test = len(dataset_full) - num_train

  dataset_train, dataset_test = torch.utils.data.random_split(dataset_full, [num_train, num_test])

  print(f'Size of the train set: {len(dataset_train)}')
  print(f'Size of the test set: {len(dataset_test)}')

  return dataset_train, dataset_test


def collate_fn(batch):
  return tuple(zip(*batch))


def create_datasets(
  root: str,
  annFile: str,
  train_ratio: float,
):
  dataset_full = CocoDataset(
    root = root,
    annFile = annFile,
    transform = get_transform(train=True),
  )

  dataset_train, dataset_test = split_dataset(dataset_full, train_ratio=train_ratio)

  return dataset_train, dataset_test


def create_dataloaders(
  dataset_train: torch.utils.data.Dataset,
  dataset_test: torch.utils.data.Dataset,
  batch_size: int,
  num_workers: int,
):
  train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    collate_fn = collate_fn,
  )

  test_dataloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers,
    collate_fn = collate_fn,
  )

  return train_dataloader, test_dataloader
