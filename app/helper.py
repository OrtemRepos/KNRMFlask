import os
from typing import (
    List, Dict, Tuple
)
import string
import json
import faiss
import pandas as pd
import numpy as np
import torch
import nltk


import knrm

class Helper:
    def __init__(self):
        self.emb_path_glove = os.environ['EMB_PATH_GLOVE']
        self.vocab_path = os.environ['VOCAB_PATH']
        self.emb_path_knrm = os.environ['EMB_PATH_KNRM']
        self.mlp_path = os.environ['MLP_PATH']
        torch.set_grad_enabled(False)

    def prepare_models(self):
        self.model = knrm.KNRM(self.emb_path_knrm, self.mlp_path)
        with open(self.vocab_path, 'r') as f:
            self.vocab = json.load(f)
        global model_is_ready
        model_is_ready = True
    
    def _handle_punctuation(self, inp_str: str) -> str:
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str
    
    def _simple_preproc(self, inp_str: str) -> str:
        base_str = inp_str.strip().lower()
        str_wo_punct = self._handle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)
    
    def prepare_index(self, documents: Dict[str, str]):
        oov_val = self.vocab['OOV']
        self.documents= documents
        idxs, docs = [], []
        for idx in documents:
            idxs.append(int(idx))
            docs.append(documents[idx])
        embeddings = []
        emb_layer = self.model.embedding.state_dict()['weight']
        for d in docs:
            tmp_emb = [self.vocab.get(w, oov_val) for w in self._simple_preproc(d)]
            tmp_emb = emb_layer[tmp_emb].mean(dim=0)
            embeddings.append(np.array(tmp_emb))
        embeddings = np.array([embedding for embedding in embeddings]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap2(self.index)
        self.index.add_with_ids(embeddings, np.array(idxs))
        index_size = self.index.ntotal
        global index_is_ready
        index_is_ready = True

        return index_size
    
    def _text_to_token_ids(self, text_list: List[str]) -> torch.FloatTensor:
        tokenized = []
        for text in text_list:
            tokenized_text = self._simple_preproc(text)
            token_idxs = [self.vocab.get(i, self.vocab['OOV']) for i in tokenized_text]
            tokenized.append(token_idxs)
        max_len = max(len(elem) for elem in tokenized)
        tokenized = [elem + [0] * (max_len - len(elem)) for elem in tokenized]
        tokenized = torch.LongTensor(tokenized)
        return tokenized
    
    def get_suggestion(self,
                       query: str, ret_k: int = 10,
                       ann_k: int = 100) -> List[Tuple[str, str]]:
        q_tokens = self._simple_preproc(query)