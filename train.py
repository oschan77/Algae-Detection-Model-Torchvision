import argparse
import torch
from data_setup import create_datasets, create_dataloaders
from model_builder import get_FasterRCNN_model
from custom_utils import save_model
from engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--annfile', type=str, required=True)
parser.add_argument('--target_path', type=str, default='saved_models')
parser.add_argument('--model_name', type=str, default='faster_rcnn_v1.pth')
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fe', type=bool, default=True)
args = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ROOT = args.root
ANNFILE = args.annfile
TARGET_PATH = args.target_path
MODEL_NAME = args.model_name
NUM_CLASSES = args.num_classes
TRAIN_RATIO = args.train_ratio
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
PRINT_FREQ = args.print_freq
LEARNING_RATE = args.lr
EPOCHS = args.epochs
FEATURE_EXTRACT = args.fe

dataset_train, dataset_test = create_datasets(
  root = ROOT,
  annFile = ANNFILE,
  train_ratio = TRAIN_RATIO,
)

train_dataloader, test_dataloader = create_dataloaders(
  dataset_train = dataset_train,
  dataset_test = dataset_test,
  batch_size = BATCH_SIZE,
  num_workers = NUM_WORKERS,
)

model = get_FasterRCNN_model(num_classes=NUM_CLASSES, feature_extract=FEATURE_EXTRACT)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
  train_one_epoch(model, optimizer, train_dataloader, DEVICE, epoch, print_freq=PRINT_FREQ)
  evaluate(model, test_dataloader, device=DEVICE)

save_model(
  model=model,
  target_path=TARGET_PATH,
  model_name=MODEL_NAME,
)
