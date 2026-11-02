import argparse
import os
from typing import Dict
from regularizers import wN3, N3
import torch
import logging
from torch import optim
from copy import deepcopy

from utils import set_seed, set_logger, get_device
from datasets import Dataset
from models import *
from optimizers import KBCOptimizer


datasets = ['MKG-W', 'MKG-Y', 'DB15K']
optimizers = ['Adagrad', 'Adam', 'AdamW']

regularizers = {
    'N3': N3,
    'wN3': wN3
}

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets))
    parser.add_argument('--regularizer', type=str, default='wN3')
    parser.add_argument('--optimizer', choices=optimizers, default='Adagrad', help="Optimizer in {}".format(optimizers))
    parser.add_argument('--max_epochs', default=200, type=int, help="Number of epochs.")
    parser.add_argument('--valid', default=5, type=float, help="Number of epochs before valid.")
    
    parser.add_argument('--rank', default=128, type=int, help="component size.")
    
    parser.add_argument('--batch_size', default=500, type=int, help="batch_size")
    parser.add_argument('--reg', default=0, type=float, help="Regularization factor")
    parser.add_argument('--init', default=1e-3, type=float, help="Initial scale")
    parser.add_argument('--learning_rate', default=1e-1, type=float, help="Learning rate")
    parser.add_argument('-weight', '--do_ce_weight', action='store_true')
    parser.add_argument('--save', action='store_true', help='Directory to save checkpoints')    
    parser.add_argument('--seed', type=int, default=2026, help="Random seed for reproducibility")

    return parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'Hit@1': h[0].item(), 'Hit@3': h[1].item(), 'Hit@10': h[2].item()}

def load_mm_features(dataset: str):
    img_features = torch.load(f"./embeddings/{dataset}-visual.pth")
    text_features = torch.load(f"./embeddings/{dataset}-textual.pth")
    return img_features, text_features

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    set_logger(args.dataset)
    device = get_device()

    logging.info(f"Using device: {device}") 
    logging.info(f"args: {args}")
    
    save_dir = os.path.join('./ckpt', args.dataset)
    if args.save:
        os.makedirs(save_dir, exist_ok=True)
    
    data_path = "./data"
    dataset = Dataset(data_path, args.dataset)
    examples = torch.from_numpy(dataset.get_train().astype('int64'))

    if args.do_ce_weight:
        ce_weight = torch.Tensor(dataset.get_weight()).to(device)
    else:
        ce_weight = None

    logging.info(f"shape: {dataset.get_shape()}")
    
    regularizer = regularizers[args.regularizer](args.reg).to(device)

    model = KGC(
        sizes=dataset.get_shape(), 
        rank=args.rank
        ).to(device)
    
    optim_method = {
        'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate),
        'AdamW': lambda: optim.AdamW(model.parameters(), lr=args.learning_rate),
    }[args.optimizer]()
    
    optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, verbose=False)

    best_valid = {}
    best_epoch = 0
    best_model = None
    
    for epoch in range(1, args.max_epochs + 1):
        cur_loss = optimizer.epoch(examples, epoch, weight=ce_weight)
        logging.info(f"Epoch {epoch}: loss = {cur_loss.item():.4f}")
        
        if epoch % args.valid == 0:
            valid = avg_both(*dataset.eval(model, 'test', -1, verbose=False))
            logging.info(f"Epoch: {epoch}, Valid MRR: {valid['MRR']:.4f}, Hits@1: {valid['Hit@1']:.4f}, Hits@3: {valid['Hit@3']:.4f}, Hits@10: {valid['Hit@10']:.4f}")
            
            if valid['MRR'] > best_valid.get('MRR', 0):
                best_valid = valid
                best_epoch = epoch
                best_model = deepcopy(model.state_dict())
                
                if args.save:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim_method.state_dict(),
                        'args': vars(args),
                        'valid': valid,
                    }, os.path.join(save_dir, 'best_checkpoint.pt'))

    logging.info(f"Best Valid MRR: {best_valid['MRR']:.4f} at epoch {best_epoch}, Hits@1:{best_valid['Hit@1']:.4f}, Hits@3: {best_valid['Hit@3']:.4f}, Hits@10: {best_valid['Hit@10']:.4f}")
    model.load_state_dict(best_model)
    logging.info(f"Loaded best model from epoch {best_epoch} for test")
    test = avg_both(*dataset.eval(model, 'test', -1, verbose=False))
    logging.info(f"Test MRR: {test['MRR']:.4f}, Hits@1: {test['Hit@1']:.4f}, Hits@3: {test['Hit@3']:.4f},Hits@10: {test['Hit@10']:.4f}")