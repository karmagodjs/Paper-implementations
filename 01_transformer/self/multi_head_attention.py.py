import math
from typing import Optional, List
import torch
from torch import nn

class PrepareForMultiHeadAttention(nn.Module):
    def __int__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        self.linear = nn.Linear(d_model, heads*d_k, bias=bias) # Linear layer of linear transform
        self.heads = heads #Number of heads
        self.d_k = d_k #Number of dimensions in vectors in each head

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x) #Linear transform
        x = x.view(*head_shape, self.heads, self.d_k) #Split last dimension into heads
        return x

class MultiHeadAttention(nn.Moduel):
    def __int__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads #Number of features per head
        self.heads = heads #Number of heads

        #These transform the query , key and value vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.softmax(dim=1) #Softmax for attention along the time dimension of key
        self.output = nn.linear(d_model, d_model) #Output layer
        self.dropout = nn.Dropout(dropout_prob) #Dropout
        self.scale = 1/ math.sqrt(self.d_k) #Scaling factor before the softmax
        self.attn = None #We store attentions so that it can be used for logging, or other computations if needed

    #Calculate scores between queries and keys
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibnd, jbhd->ijbh', query, key) #Calculate QK^T

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[0] == key_shape[0]
        assert mask.shape[0] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1) #Same mask applied to all heads
        return mask #resulting mask has shape [seq_len_q, seq_len_k, batch_size, heads]

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query_shape, key_shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores *= self.scale

        #Apply Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores) #Softmax attn
        tracker.debug('attn', attn) # save attn if debugg
        attn = self.dropout(attn) #Apply Dropout

        x = torch.einsum('ijbh,jbhd->ibhd', attn, value) #Multiply by values
        self.attn = attn.detach() #Save attentions for any other calculations
        x = x.reshape(seq_len, batch_size, -1) #Concatenate multiple heads

        return self.output(x) #Output layer