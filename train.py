import argparse
import torch
from data_setup import create_datasets, create_dataloaders
from model_builder import get_FasterRCNN_model
from custom_utils import save_model
from engine import train_one_epoch, evaluate


def parse_args():
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
    parser.add_argument('--fe', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train, dataset_test = create_datasets(
        root=args.root,
        annFile=args.annfile,
        train_ratio=args.train_ratio,
    )

    train_dataloader, test_dataloader = create_dataloaders(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_FasterRCNN_model(
        num_classes=args.num_classes,
        feature_extract=args.fe,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
        )
        evaluate(model=model, data_loader=test_dataloader, device=device)

    save_model(
        model=model,
        target_path=args.target_path,
        model_name=args.model_name,
    )


if __name__ == '__main__':
    main()
