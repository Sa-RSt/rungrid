# Sample main.py file from rungrid template
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, NoReturn

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

from rungrid.experiment import (
    STATE_ARG_RECORD,
    Experiment,
    Sampler,
    Trial,
    VarNamespace,
    step_method,
)
from rungrid.version import version_from_most_recent_mtime


class MNISTTuning(Experiment):
    def variables(self, v: VarNamespace, s: Sampler) -> None:
        v.lr = s.log_uniform(1e-6, 1e-1)
        v.wd = s.log_uniform(1e-6, 1e-1)
        v.layers = s.integer(1, 6)
        v.activations = s.list(s.categorical(["relu", "leaky_relu", "selu"]), v.layers)
        v.kernel_size = 3
        v.epochs = 5

    def version(self) -> str:
        return version_from_most_recent_mtime(type(self))

    def first_step(self) -> Callable:
        return self.step_initialize_training

    @lru_cache
    @staticmethod
    def get_datasets() -> tuple[Dataset, Dataset]:
        train_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        test_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        return train_dataset, test_dataset

    @step_method(
        cache=True,  # "deterministic" initialization
        recover_from=[KeyboardInterrupt, ArithmeticError],
    )
    def step_initialize_training(self, trial: Trial) -> NoReturn:
        seq = []
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "selu": nn.SELU,
        }
        for activation in trial.v.activations:
            seq.append(nn.LazyConv2d(out_channels=64, kernel_size=trial.v.kernel_size))
            ac = activations[activation]
            seq.append(ac())
        seq.append(nn.Flatten())
        seq.append(nn.LazyLinear(out_features=10))
        model = self.to_device(nn.Sequential(*seq))
        self.exit_next_step(
            self.step_one_epoch,
            last=model,
            best=None,
            best_score=float("inf"),
            epoch=0,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.AdamW(
                model.parameters(), lr=trial.v.lr, weight_decay=trial.v.wd
            ),
        )

    def to_device(self, x):
        if torch.cuda.is_available():
            return x.cuda()
        return x.cpu()

    @step_method(
        cache=False,
        recover_from=[KeyboardInterrupt, ArithmeticError],
    )
    def step_one_epoch(
        self,
        trial: Trial,
        last: nn.Module,
        best: nn.Module | None,
        best_score: Annotated[float, STATE_ARG_RECORD],
        epoch: Annotated[int, STATE_ARG_RECORD],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> NoReturn:
        train_dataset = type(self).get_datasets()[0]
        if epoch > trial.v.epochs:
            self.exit_next_step(self.step_score, model=best)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        train_loss = 0.0

        last = self.to_device(last)
        best = self.to_device(best) if best is not None else None
        for inputs, labels in train_loader:
            inputs, labels = self.to_device(inputs), self.to_device(labels)

            optimizer.zero_grad()

            outputs = last(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if train_loss < best_score:
            best = last
            best_score = train_loss

        print("epoch", epoch, f"loss: {train_loss:.5f}")
        epoch += 1

        self.exit_next_step(self.step_one_epoch, fill_locals=True)

    @step_method(cache=False)
    def step_score(
        self, trial: Trial, model: Annotated[nn.Module, STATE_ARG_RECORD]
    ) -> float:
        test_dataset = type(self).get_datasets()[1]
        model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = self.to_device(images), self.to_device(labels)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"{accuracy=:.2f}%")
        return accuracy
