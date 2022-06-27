import torch
import os
import copy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.optim import SGD, Adam
from backbone.ResNet18 import resnet18
from backbone.MNISTMLP import MNISTMLP
from torch.autograd import Variable
import torch.nn as nn
from copy import deepcopy
from utils.conf import get_device


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' meta-consolidation over graph hyper networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class monologue(ContinualModel):
    NAME = 'monologue'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(monologue, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.fine_tune = args.fine_tune
        self.fine_tuned = self.net.architecture
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.task_aggregate = torch.zeros(5 if self.args.dataset != 'seq-tinyimg' else 10).to(self.device) if args.consolidate else None
        self.task_embedding = torch.zeros(5 if self.args.dataset != 'seq-tinyimg' else 10).to(self.device) if args.consolidate else None
        self.node_embeds = []
        self.precision_matrices = []

    def begin_task(self, dataset):
        if self.args.consolidate:
            self.task_embedding = torch.zeros(5 if self.args.dataset != 'seq-tinyimg' else 10).to(self.device)
            self.task_embedding[self.current_task] += 1

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs, embeds = self.net(inputs, return_embeddings=True, task_embedding=self.task_embedding)

        loss = self.loss(outputs, labels)
        importance = 0.1

        # if self.args.consolidate:
        #     for i in range(len(self.node_embeds)):
        #         loss += importance * torch.nn.functional.mse_loss(self.precision_matrices[i]*embeds, self.precision_matrices[i]*self.node_embeds[i])

        # if self.args.consolidate:
        #     if self.node_embeds:
        #         loss += importance * torch.nn.functional.mse_loss(self.precision_matrices[-1]*embeds, self.precision_matrices[-1]*self.node_embeds[-1])

        loss.backward()
        if len(self.precision_matrices) == self.current_task:
            self.precision_matrices.append(torch.zeros_like(embeds))

        self.precision_matrices[-1] += embeds.grad**2
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset) -> None:
        self.current_task += 1
        if self.args.consolidate:
            self.task_aggregate += self.task_embedding

        embeds = self.net(return_embeddings=True, task_embedding=self.task_aggregate)  # predict all parameters of architecture
        if self.args.consolidate:
            self.node_embeds.append(embeds.detach())

        if self.args.dataset in ['seq-mnist', 'perm-mnist', 'rot-mnist']:
            unfine_tuned = MNISTMLP(28 * 28, 10)
        elif self.args.dataset == 'seq-cifar10':
            unfine_tuned = resnet18(10)
        else:
            unfine_tuned = resnet18(200)

        unfine_tuned.to(self.device)
        unfine_tuned.load_state_dict(copy.deepcopy(self.net.architecture.state_dict()))

        if self.fine_tune:
            fine_tune_opt = SGD(unfine_tuned.parameters(), lr=self.args.fine_tune_lr)
            buf_inputs, buf_labels = self.buffer.get_all_data(transform=self.transform)
            for i in range(self.args.fine_tune_epochs):
                for j in range(self.args.buffer_size//self.args.minibatch_size):
                    fine_tune_opt.zero_grad()
                    outputs = unfine_tuned(buf_inputs[j*self.args.minibatch_size:(j+1)*self.args.minibatch_size])
                    loss = self.loss(outputs, buf_labels[j*self.args.minibatch_size:(j+1)*self.args.minibatch_size])
                    loss.backward()
                    fine_tune_opt.step()

        self.fine_tuned = unfine_tuned

        model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fine_tuned(x)
