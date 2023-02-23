import argparse
import torch
import torch.nn.parallel
import torch.distributed as dist
from data_setup import create_datasets, create_dataloaders
from model_builder import get_faster_rcnn_model
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

    # Initialize distributed training environment
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl', init_method='env://')

    dataset_train, dataset_test = create_datasets(
        root=args.root,
        annFile=args.annfile,
        train_ratio=args.train_ratio,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=torch.cuda.device_count(),
        rank=dist.get_rank() if torch.cuda.device_count() > 1 else 0,
        shuffle=True,
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_test,
        num_replicas=torch.cuda.device_count(),
        rank=dist.get_rank() if torch.cuda.device_count() > 1 else 0,
        shuffle=False,
    )

    train_dataloader, test_dataloader = create_dataloaders(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
    )

    model = get_faster_rcnn_model(
        num_classes=args.num_classes,
        feature_extract=args.fe,
    )

    # Wrap the model with DistributedDataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            output_device=dist.get_rank(),
        )
    else:
        model = torch.nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        _, train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
        )

        print(f'train_loss: {train_loss}')

        test_sampler.set_epoch(epoch)
        evaluate(model=model, data_loader=test_dataloader, device=device)

    save_model(
        model=model,
        target_path=args.target_path,
        model_name=args.model_name,
    )

    # Clean up the distributed training environment
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()