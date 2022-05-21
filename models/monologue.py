import torch
import os
import copy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.optim import SGD, Adam
from backbone.ResNet18 import resnet18
from backbone.MNISTMLP import MNISTMLP


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' meta-consolidation over graph hyper networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class monologue(ContinualModel):
    NAME = 'monologue'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(monologue, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.fine_tune = args.fine_tune
        self.fine_tuned = [self.net.architecture]
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.task_aggregate = torch.zeros(5 if self.args.dataset != 'seq-tinyimg' else 10).to(self.device) if args.consolidate else None
        self.task_embedding = torch.zeros(5 if self.args.dataset != 'seq-tinyimg' else 10).to(self.device) if args.consolidate else None
        self.node_embeds = []

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

        # distribution = torch.distributions.multivariate_normal.MultivariateNormal(self.task_embedding, 0.001*torch.eye(len(self.task_embedding)).to(self.device))
        outputs, embeds = self.net(inputs, return_embeddings=True, task_embedding=self.task_embedding)

        loss = self.loss(outputs, labels)

        if self.args.consolidate:
            for i in range(len(self.node_embeds)):
                loss += torch.nn.functional.mse_loss(embeds, self.node_embeds[i]) / 30

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset) -> None:
        self.fine_tuned = []
        self.current_task += 1
        if self.args.consolidate:
            self.task_aggregate += self.task_embedding

        for _ in range(100):
            # distribution = torch.distributions.multivariate_normal.MultivariateNormal(self.task_aggregate/self.current_task,
            #                                                                           0.001 * torch.eye(
            #                                                                               len(self.task_embedding)).to(
            #                                                                               self.device))
            embeds = self.net(return_embeddings=True, task_embedding=self.task_aggregate/self.current_task)  # predict all parameters of architecture

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

        self.fine_tuned.append(unfine_tuned)

        model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros([len(x), 10]).to(self.device)
        for i in range(len(self.fine_tuned)):
            _, pred = torch.max(self.fine_tuned[i](x).data, 1)
            out[torch.arange(len(x)), pred] += 1
        return out
