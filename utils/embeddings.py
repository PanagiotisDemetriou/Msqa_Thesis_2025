import torch
import torch.nn as nn
import torch.nn.functional as F

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



class Text_Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super(Text_Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)
    
class Text_Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}

    def tokenize(self, text):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text.split()]

    def detokenize(self, indices):
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices])
    
### test tokenizer ###
if __name__ == "__main__":
      vocab = ['<pad>', '<unk>', 'hello', 'world', 'unknown']
      tokenizer = Text_Tokenizer(vocab)
      text = "hello unknown world"
      tokens = tokenizer.tokenize(text)
      print("Tokens:", tokens)
      detokenized = tokenizer.detokenize(tokens)
      print("Detokenized:", detokenized)
   
      