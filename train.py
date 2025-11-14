import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer


# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_func(config):
    """Training function to be distributed across workers."""

    # Get training configuration
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 0.001)
    epochs = config.get("epochs", 5)

    # Prepare datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

     train_dataset = datasets.FashionMNIST(
        root="/tmp/data",
        train=True,
        download=True,
        transform=transform
    )

     test_dataset = datasets.FashionMNIST(
        root="/tmp/data",
        train=False,
        download=True,
        transform=transform
    )

    # Prepare data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

     test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Prepare model, optimizer, and data loaders for distributed training
    model = Net()
    model = train.torch.prepare_model(model)

     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

     train_loader = train.torch.prepare_data_loader(train_loader)
    test_loader = train.torch.prepare_data_loader(test_loader)

    # Training loop
    print("Starting the training loop!")
    for epoch in range(epochs):
        print(f"Starting on epoch {epoch}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

         for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

             train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)

             if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

         with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)

         train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total

         print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")

        # Report metrics and checkpoint
        checkpoint = None
        if train.get_context().get_world_rank() == 0:
            checkpoint = train.torch.TorchCheckpoint.from_state_dict(model.state_dict())

         train.report(
            metrics={
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            },
            checkpoint=checkpoint
        )

     print("Training completed!")


  def main():
    """Main function to setup and run Ray Train."""

    # Configure S3/Minio settings
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["AWS_ENDPOINT_URL"] = "http://minio.minio.svc.cluster.local:9000"

    # For s3fs (used by Ray for S3 storage)
    os.environ["S3_ENDPOINT_URL"] = "http://minio.minio.svc.cluster.local:9000"

    # Initialize Ray
    ray.init()

    # Configure the trainer
    scaling_config = ScalingConfig(
        num_workers=2,  # Number of distributed training workers
        use_gpu=True,  
        resources_per_worker={
            "CPU": 2,
            "GPU": 1
        }
    )

     run_config = RunConfig(
        name="fashion_mnist_train",
        storage_path="s3://checkpoints/",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="test_acc",
            checkpoint_score_order="max"
        )
    )

    # Create the trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "batch_size": 64,
            "lr": 0.001,
            "epochs": 5
        },
        scaling_config=scaling_config,
        run_config=run_config
    )

    # Run training
    result = trainer.fit()

    print("\n" + "="*50)
    print("Training Results:")
    print("="*50)
    print(f"Best checkpoint: {result.checkpoint}")
    print(f"Best metrics: {result.metrics}")
    print("="*50)


  if __name__ == "__main__":
    main()