import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Cross_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads,k_d=None,v_d=None, dropout=0.1,batch_first=True):
        super(Cross_Attention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,batch_first=batch_first, kdim=k_d, vdim=v_d)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        ### does it need a feed forward layer here? ###

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_output_weights = self.multihead_attention(query, key, value, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(query + attn_output)
        return output
    
### Example usage? How to test it? ###




# Text Tokenizer into subwords
class Text_Tokenizer:
    def __init__(self):
      self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def tokenize(self, text):
        # Maps each word to its index
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, indices):
        # Maps each index back to its word
        return self.tokenizer.decode(indices)

    
   
   
### test tokenizer ###
if __name__ == "__main__":
      tokenizer = Text_Tokenizer()
      text = "Testing tokenization and detokenization."
      tokens = tokenizer.tokenize(text)
      print("Tokens:", tokens)
      detokenized = tokenizer.detokenize(tokens)
      print("Detokenized:", detokenized)
   
      