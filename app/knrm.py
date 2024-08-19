from typing import Dict, List


import torch
import torch.nn.functional as F

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )
    
class KNRM(torch.nn.Module):
    def __init__(self,
                 emb_path: str,
                 mlp_path: str,
                 kernel_num: int = 21,
                 sigma: float = 0.1,
                 exact_sigma: float = .001,
                 out_layers: List[int] = []
                 ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.load(emb_path)['weight'],
            freeze=True,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.mlp._load_state_dict(torch.load(mlp_path))
        self.out_activation = torch.nn.Sigmoid()

    
    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (self.kernel_num - 1) - 1.
            sigma = self.sigma
            if mu > 1.:
                sigma = self.exact_sigma
                mu = 1.
            kernels.append(GaussianKernel(mu, sigma))
        return kernels
    
    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)
    
    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        embed_query = self.embeddings(query.long())
        embed_doc = self.embeddings(doc.long())

        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix
    
    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)
        
        kernels_out = torch.stack(KM, dim=-1)
        return kernels_out
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']

        matching_matrix = self._get_matching_matrix(query, doc)

        kernels_out = self._apply_kernels(matching_matrix)

        out = self.mlp(kernels_out)
        return out