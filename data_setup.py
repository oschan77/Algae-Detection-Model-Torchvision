import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from typing import Tuple, List

# CocoDataset class that extends CocoDetection class
class CocoDataset(CocoDetection):
    # getitem method that returns transformed image and target
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        # get original image and target from CocoDetection class
        img, ori_target = super().__getitem__(index)

        # Extract boxes, labels, area, and iscrowd from target
        num_objs = len(ori_target)
        boxes = [[max(0, ori_target[i]['bbox'][0]), max(0, ori_target[i]['bbox'][1]), 
                  min(4908, ori_target[i]['bbox'][0] + ori_target[i]['bbox'][2]), 
                  min(3264, ori_target[i]['bbox'][1] + ori_target[i]['bbox'][3])] 
                  for i in range(num_objs)]
        labels = [ori_target[i]['category_id'] for i in range(num_objs)]
        area = [ori_target[i]['area'] for i in range(num_objs)]
        iscrowd = [ori_target[i]['iscrowd'] for i in range(num_objs)]

        # Create target dictionary with boxes, labels, image_id, area, and iscrowd
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                  'labels': torch.as_tensor(labels, dtype=torch.int64),
                  'image_id': torch.tensor([ori_target[0]['image_id']]),
                  'area': torch.as_tensor(area, dtype=torch.float32),
                  'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)}
        return img, target

# Helper function to get the appropriate transformation based on the train flag
def get_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])

# Function to split dataset into train and test datasets
def split_dataset(dataset_full: torch.utils.data.Dataset, train_ratio: float) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    num_train = int(len(dataset_full) * train_ratio)
    num_test = len(dataset_full) - num_train

    # Randomly split the dataset into train and test datasets
    dataset_train, dataset_test = torch.utils.data.random_split(dataset_full, [num_train, num_test])
    return dataset_train, dataset_test

# Function to collate a batch of samples into a single batch
def collate_fn(batch: List[Tuple[torch.Tensor, dict]]) -> Tuple[torch.Tensor, List[dict]]:
    return tuple(zip(*batch))

# Function to create train and test datasets from root directory and annotation file
def create_datasets(root: str, annFile: str, train_ratio: float) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    dataset_full = CocoDataset(root=root, annFile=annFile, transform=get_transform(train=True))
    dataset_train, dataset_test = split_dataset(dataset_full, train_ratio=train_ratio)
    return dataset_train, dataset_test

# Function to create train and test dataloaders from train and test datasets
def create_dataloaders(dataset_train: torch.utils.data.Dataset, dataset_test: torch.utils.data.Dataset, batch_size: int, num_workers: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, test_dataloader