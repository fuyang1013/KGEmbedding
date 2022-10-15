"""
author: flyangovoyang@163.com
first create: 2022-10-15
last updated: 2022-10-15
"""
from calendar import c
import os
import pdb
from argparse import ArgumentParser
import logging
import numpy as np
import time
import torch
from torch import sigmoid, pow
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s')


class Triplet:
    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t


class Example:
    def __init__(self, triplet, score):
        self.triplet = triplet
        self.score = score


def loss_function(truth_score, pred_score):
    """
    params:
        truth_score: [batch_size], float
        pred_score: [batch_size], float
    return:
        loss: float
    """
    loss_fn = torch.nn.MSELoss()
    return loss_fn(pred_score, truth_score)


class UKGE(torch.nn.Module):
    """implementation of UKGE model, for more details see https://arxiv.org/abs/1811.10667
    """
    def __init__(self, num_ents, num_rels, num_dim, mapping='logistic'):
        """
        params:
            num_ents: 实体数量
            num_rels: 关系数量
            num_dim: 嵌入维度
            mapping: 
        """
        super().__init__()
        self.ent_emb = torch.nn.Embedding(num_ents, num_dim)
        self.rel_emb = torch.nn.Embedding(num_rels, num_dim)
        self.w = torch.nn.Parameter(torch.FloatTensor([0]))
        self.b = torch.nn.Parameter(torch.FloatTensor([0]))
        assert mapping == 'logistic' or mapping == 'bounded_rectifier'
        if mapping == 'logistic':
            self.score_function = self.logistic
        else:
            self.score_function = self.bounded_rectifier
    
    def logistic(self, x):
        return torch.div(1, 1 + pow(np.e,  - (self.w * x + self.b)))
    
    def bounded_rectifier(self, x):
        return torch.min(torch.max(self.w * x + self.b, 0), 1)

    def forward(self, h, r, t, scores=None):
        """
        params:
            h: [batch_size, dim]
            r: [batch_size, dim]
            t: [batch_size, dim]
        return:
            score: float
        """
        h = self.ent_emb(h)
        t = self.ent_emb(t)
        r = self.rel_emb(r)

        x = (h * t * r).sum(dim=-1)
        preds = self.score_function(x)
        if r is not None:
            loss = loss_function(preds, scores)
            return preds, loss
        return preds


def read_file(infile, num_examples=None, verbose=False):
    examples = []
    for line in open(infile):
        ll = line.rstrip().split('\t')
        h, r, t, s = ll
        h, r, t, s = int(h), int(r), int(t), float(s)
        triplet = Triplet(h, r, t)
        examples.append(Example(triplet, s))

        if num_examples is not None and len(examples) >= num_examples:
            break
    return examples


def build_dataloader(examples, batch_size, sample='random'):
    heads = torch.LongTensor([x.triplet.h for x in examples])
    tails = torch.LongTensor([x.triplet.t for x in examples])
    relations = torch.LongTensor([x.triplet.r for x in examples])
    scores = torch.FloatTensor([x.score for x in examples])
    dataset = TensorDataset(heads, relations, tails, scores)
    sampler = RandomSampler(dataset) if sample == 'random' else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def evaluate(model, eval_dataloader, args, verbose=False):
    device = torch.device(args.device)
    
    total_loss = 0
    for batch in eval_dataloader:
        heads, relations, tails, scores = tuple([t.to(device) for t in batch])
        _, loss = model(h=heads, r=relations, t=tails, scores=scores)
        total_loss += loss.item()
    
    return total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0


def train(model: UKGE, args):
    device = torch.device(args.device)
    logging.info(f'device: {args.device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_examples = 200 if args.debug else None
    train_examples = read_file(args.train_file, num_examples)
    eval_examples = read_file(args.eval_file, num_examples)

    train_dataloader = build_dataloader(train_examples, args.batch_size, 'random')
    eval_dataloader = build_dataloader(eval_examples, args.batch_size, 'sequential')
    logging.info(f'train examples: {len(train_examples)}, eval_examples: {len(eval_examples)}')

    global_steps = 0
    logging.info('start to train...')
    for epoch in range(args.train_epochs):
        for batch in train_dataloader:
            global_steps += 1
            optimizer.zero_grad()
            heads, relations, tails, scores = tuple([t.to(device) for t in batch])
            _, loss = model(h=heads, r=relations, t=tails, scores=scores)
            loss.backward()
            optimizer.step()

            if (global_steps + 1) % args.display_steps == 0 or (global_steps + 1) % len(train_dataloader) == 0 or args.debug:
                logging.info(f'epoch={epoch}, steps={global_steps}, loss={loss.item():.4f}')
        
        logging.info('evaluating...')
        eval_loss = evaluate(model, eval_dataloader, args, verbose=False)
        logging.info(f'epoch={epoch}, mse={eval_loss:.4f}')
        
        save_model_name = f'{args.save_model_name}-epoch-{epoch}-mse-{eval_loss:.4f}'
        if not os.path.exists(os.path.join(args.save_model_dir, save_model_name)):
            os.mkdir(os.path.join(args.save_model_dir, save_model_name))

        save_model_path = os.path.join(args.save_model_dir, save_model_name, 'model.pt')
        torch.save(model.state_dict(), save_model_path)
        open(os.path.join(args.save_model_dir, save_model_name, 'config.json'), 'w').write(str(args))

        if args.debug:
            break


def init_model(args):
    model = UKGE(num_ents=args.num_ents, num_rels=args.num_rels, num_dim=args.num_dim, mapping=args.mapping)
    if args.checkpoint is not None:
        model_path = os.path.join(args.checkpoint, 'model.pt')
        model.load_state_dict(torch.load(model_path))
    device = torch.device(args.device)
    model = model.to(device)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--save_model_dir', type=str, default='model_save')
    parser.add_argument('--save_model_name', type=str, default=time.strftime('%m%d_%H%M%S', time.localtime()))
    parser.add_argument('--lr', type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--display_steps', type=int, default=200)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num_ents', type=int)
    parser.add_argument('--num_rels', type=int)
    parser.add_argument('--num_dim', type=int)
    parser.add_argument('--mapping', type=str, choices=['logistic', 'bounded_rectifier'])
    args = parser.parse_args()

    model = init_model(args)
    if args.do_train:
        train(model, args)
    elif args.do_eval:
        num_examples = 200 if args.debug else None
        eval_examples = read_file(args.eval_file, num_examples)
        eval_dataloader = build_dataloader(eval_examples, args.batch_size, 'sequential')
        print(evaluate(model, eval_dataloader, args, verbose=True))
    else:
        logging.info('unknown task')


