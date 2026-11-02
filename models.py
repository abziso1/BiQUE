from torch import nn
import torch
from typing import Tuple, List, Dict
from qutils import *
from abc import ABC, abstractmethod
from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1,
            verbose: bool = True
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex', disable=not verbose) as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())] + [queries[b_begin + i, 2].item()]
                        scores[i, torch.as_tensor(filter_out, dtype=torch.long, device=scores.device)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

class BiQUEScorer(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, head, relation, all):
        rel_add = relation[:, self.rank * 8:]
        rel_mul = relation[:, :self.rank * 8]
        
        head += rel_add
        
        w_a, x_a, y_a, z_a = torch.chunk(head, 4, dim=-1)
        w_b, x_b, y_b, z_b = torch.chunk(rel_mul, 4, dim=-1)
        
        A = complex_mul(w_a,w_b) - complex_mul(x_a,x_b) - complex_mul(y_a,y_b) - complex_mul(z_a,z_b)
        B = complex_mul(w_a,x_b) + complex_mul(x_a,w_b) + complex_mul(y_a,z_b) - complex_mul(z_a,y_b)
        C = complex_mul(w_a,y_b) - complex_mul(x_a,z_b) + complex_mul(y_a,w_b) + complex_mul(z_a,x_b)
        D = complex_mul(w_a,z_b) + complex_mul(x_a,y_b) - complex_mul(y_a,x_b) + complex_mul(z_a,w_b)
        
        res = torch.cat([A,B,C,D], dim=-1)
        score = res @ all.T
        return score

class KGC(KBCModel):
    def __init__(
        self, sizes: Tuple[int, int, int], rank: int
    ):

        super().__init__()

        self.rank = rank
        nentity, nrelation, _ = sizes

        self.entity_embeddings = nn.Embedding(nentity, rank * 8)
        self.relation_embeddings = nn.Embedding(nrelation, rank * 16)
        self.scorer = BiQUEScorer(rank)      
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)       
    
    def forward(self, x):
        h_ids = x[:, 0]
        r_ids = x[:, 1]
        t_ids = x[:, 2]
        
        all = self.entity_embeddings.weight
        head = self.entity_embeddings(h_ids)
        relation = self.relation_embeddings(r_ids)
        tail = self.entity_embeddings(t_ids)
        score = self.scorer(head, relation, all)
        factors = [(get_norm(head, 8), get_norm(relation[:, :self.rank*8], 8), get_norm(tail, 8))]

        score_final = (
            score
        )
        
        return score_final, factors
