import torch, torch.nn as nn
import tiktoken
import time

import numpy as np

from torch.utils.data import DataLoader, Dataset
from gpt_download3 import download_and_load_gpt2

GPT_CONFIG_124M = {
    'vocab_size' : 50257,
    'context_length' : 1024,
    'emb_dim' : 768,
    'n_heads' : 12,
    'n_layers' : 12,
    'drop_rate' : 0.1,
    'qkv_bias' : False
}

# method to convert text into tokens -> tensor
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text) # [1,2,3]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # list - > tensor; [ [1,2,3] ] (1 x n) 
    return encoded_tensor

# method to convert back tokens into text -> list
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 1 x n -> n
    return tokenizer.decode(flat.tolist()) # tensor -> list

# given the indices produce the max new tokens output by the model (no print in built code) -> predicted tokens
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # take these much indexes to generate (max_new_tokens)

        with torch.no_grad(): # stop calculating gradients and computing loss
            logits = model(idx_cond) # get the weights

        logits = logits[:, -1, :] # (batch size x seq len x vocab size) -> (batch_size x vocab size)
        probas = torch.softmax(logits, dim = -1) # get the probabilities
        idx_next = torch.argmax(probas, dim = -1, keepdim = True) # get the highest probability token index
        idx = torch.cat((idx, idx_next), dim = 1) # concatenate the output index to the previous input index for 'new' input indixcs for the next prediction
    
    return idx

# create input and target pairs with specified batch size -> [(input, output) in batches]
def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2") # bpe tokenizer
    dataset = GPTDatasetV1(txt, tokenizer = tokenizer, max_length = max_length, stride = stride)

    dataloader = DataLoader(dataset, batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)
    # drop last = last the remaining dataset if they couldn't fit in the batch
    # num workers = parallel workers 
    return dataloader

# take the batch of input and target pairs and calculate its loss -> loss
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch) # the result should be predicted output weights
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) # do the cross entropy loss of (input, target)
    return loss

# get the total loss per batch -> loss per batch
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0

    # data loader could be training set loader or validation set loader

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches: # calc loss until the i reaches upto the num batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches # get the loss per batch

# the training would happen first each iteration and the respective output could be seen by "generate_and_print_sample" function 
# the weights get trained and this "start_context" is the input to the model which would have its own weight and embeddings different from other contexts
# the more the training happens each iteration the predicted tokens will be generated differently  
# -> train loss, val loss, tokens seen
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # stop calculating gradients for now
            loss = calc_loss_batch(input_batch, target_batch, model, device) # -> loss
            loss.backward() # -> back propagate
            optimizer.step() # -> take step size towards the optimal solution
            tokens_seen += input_batch.numel() 
            global_step += 1
        
            if global_step % eval_freq == 0: # -> to ensure that this block of code will be triggered after this much iteration (eval_freq)
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter) # -> batch loss of train and val loader
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # print(f"ep {epoch + 1} (step {global_step:06d})", 
                #       f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")
                
        generate_and_print_sample(model, tokenizer, device, start_context) # creates the predicted texts

    return train_losses, val_losses, track_tokens_seen

# gives the batch loss of train and val data dataset
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter) # -> loss per eval_iter (batch size)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)

    model.train()
    return train_loss, val_loss

# to return the predicted tokens during loss calculation
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval() # -> in evaluation mode : dropout layers are switched off, batch uses accumulated mean and variance 
    context_size = model.position_embeding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer = tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model = model, idx = encoded, max_new_tokens = 50, context_size = context_size) # -> predicted token ids 
        decoded_text = token_ids_to_text(token_ids = token_ids, tokenizer = tokenizer) # -> get the text
        # print(decoded_text.replace("\n", " "))
        model.train() # -> allow dropuouts again and batch normalization to calculate mean and variance different on each iteration

# top k sampling + temperature scaling to introduce the technique to generate the output -> new predicted indices
def generate(model, idx, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim = 1)
    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.position_embeding.weight = assign(gpt.position_embeding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_query.weight = assign(
            gpt.trf_blocks[b].att.w_query.weight, q_w.T)
        gpt.trf_blocks[b].att.w_key.weight = assign(
            gpt.trf_blocks[b].att.w_key.weight, k_w.T)
        gpt.trf_blocks[b].att.w_value.weight = assign(
            gpt.trf_blocks[b].att.w_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_query.bias = assign(
            gpt.trf_blocks[b].att.w_query.bias, q_b)
        gpt.trf_blocks[b].att.w_key.bias = assign(
            gpt.trf_blocks[b].att.w_key.bias, k_b)
        gpt.trf_blocks[b].att.w_value.bias = assign(
            gpt.trf_blocks[b].att.w_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim']) # 50257 x 768
        self.position_embeding = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 1024 x 768
        self.drop_embedding = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])] # *[] -> unpacks all 12 layers of transformer output 
        )

        self.final_norm = LayerNorm(cfg['emb_dim']) # (1 x 768) returns normalized weights with trainable distribution
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False) # apply affine transformation

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape # (batch size x input tokens length)
        tok_embeds = self.token_embedding(in_idx) # goes to forward method and give back the token embeddings 
        pos_embeds = self.position_embeding(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds # add both token and positional embeddings 
        x = self.drop_embedding(x)
        x = self.trf_blocks(x) 
        x = self.final_norm(x) # perform normalization 
        logits = self.out_head(x) # probability scores of token that could be predicted next
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'], # 768
            d_out = cfg['emb_dim'], # 768
            context_length = cfg['context_length'], # 1024
            num_heads = cfg['n_heads'], # 12
            dropout = cfg['drop_rate'], # 0.1 
            qkv_bias = cfg['qkv_bias']) # False
        
        self.ff = FeedForwad(cfg) # induces a neural network (1024 -> 1024 x 4 -> GELU -> 1024 x 4 -> 1024)
        self.norm1 = LayerNorm(cfg['emb_dim']) # 1 x 768 -> normalized weights of emb dim
        self.norm2 = LayerNorm(cfg['emb_dim']) 
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x # shortcut is output which we add back to the input so that it doesn't make our gradients vanish
        x = self.norm1(x) # applying 1st normalization on the embeddings
        x = self.att(x) # multi head attention mechanism -> x becomes context vector
        x = self.drop_shortcut(x) # drop some weights to 0
        x = x + shortcut # here add the output back to the input 'shortcut' 

        shortcut = x # assume new input = shorcut
        x = self.norm2(x) # apply 2nd normalization on the embeddings
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # again add the output back to the input 'shortcut'

        return x  # return the output tensor 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads == 0)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5

        # Parameter -> trainable weights 
        self.scale = nn.Parameter(torch.ones(emb_dim))  # all the 1's of 768 dim is trainable 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # all the 0's of 768 dim is trainable 
        # scale and shift allows the model to create their own distribution so that normalization doesn't force it to have same distribution all the time  

    def forward(self, x):
        # x -> (1 x 768)
        mean = x.mean(dim = -1, keepdim = True) # keepdim -> preserve dimension after operation
        var = x.var(dim = -1, keepdim = True, unbiased = False) # unbiased -> take the whole sample space 'n' not sample size that is 'n - 1' doesn't applies Bessel's correction
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # z = x - x' / sqrt(var + some_weight) -> normalized weights
        return self.scale * norm_x + self.shift # after norm return the (1 x 768) values with scaled and shifted values -> adjusting the distribution accordingly during training
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForwad(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(), # for -ve values estimate it as lower to 0 but not exactly 0 unlike RELU function
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = [] # [token list of inputs] 
        self.target_ids = [] # [respective list of outputs]

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride): # (0, total token ids - max token ids that will be predicted, stride -> step size)

            # len(token_ids) - max_length -> gives upper bound to 'i'
            # for last i value i.e. i = len(token_ids) - max_length - 1
            # input_chunk = token_ids[len(token_ids) - max_length - 1, (len(token_ids) - max_length - 1 + max_length) -> len(token_ids) - 1]
            # output_chunk = token_ids[(len(token_ids) - max_length - 1 + 1) -> len(token_ids) - max_length, (len(token_ids) - max_length + max_length + 1) -> len(token_ids) + 1]

            input_chunk = token_ids[i : i + max_length] # "every effort moves you" -> input data
            target_chunk = token_ids[i + 1 : i + max_length + 1] # "effort moves you forward" -> output data

            self.input_ids.append(torch.tensor(input_chunk)) # [ [i1], [i2], [i3] ]
            self.target_ids.append(torch.tensor(target_chunk)) # [ [t1], [t2], [t3] ]

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index] # [xi], [yi]

# for reproducibility to generate same random weights
torch.manual_seed(123)

start_time = time.time()

# model = GPTModel(GPT_CONFIG_124M) # model instance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings, params = download_and_load_gpt2(model_size = "124M", model_dir = "gpt2")

# dummy dataset
with open("the-verdict.txt", 'r', encoding = "utf-8")as f:
    story_book = f.read()

# bpe tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# token_ids = generate_text_simple(model = model, idx = text_to_token_ids(story_book, tokenizer),
#                         max_new_tokens = 10, context_size = GPT_CONFIG_124M["context_length"])

total_tokens = len(tokenizer.encode(story_book))

# cross validation
train_ratio = 0.90
split_idx = int(train_ratio * total_tokens)

# datas
train_data = story_book[: split_idx]
val_data = story_book[split_idx :]

# dataloaders 
train_loader = create_dataloader_v1(train_data, batch_size = 2, max_length = GPT_CONFIG_124M["context_length"], stride = GPT_CONFIG_124M["context_length"], drop_last = True, shuffle = True, num_workers = 0)
val_loader = create_dataloader_v1(val_data, batch_size = 2, max_length = GPT_CONFIG_124M["context_length"], stride = GPT_CONFIG_124M["context_length"], drop_last = False, shuffle = False, num_workers = 0)

# token_ids = generate(model = model, idx = text_to_token_ids("every effort moves you", tokenizer), max_new_tokens = 25, context_size = GPT_CONFIG_124M["context_length"], top_k = 25, temperature = 1.4)
# print(f"output text :\n {token_ids_to_text(token_ids, tokenizer)}")

# if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
#     print("not enough tokens for the training loader")

# if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
#     print("not enough tokens for the validation loader")

# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)
#     print(train_loss, val_loss)

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
gpt = GPTModel(NEW_CONFIG).to(device)
gpt.eval()

# optimizer
optimizer = torch.optim.AdamW(gpt.parameters(), lr = 0.0004, weight_decay = 0.1)

# iteration for training
num_epochs = 10

# load the pre-trained weights into the model
load_weights_into_gpt(gpt, params)
# gpt.to(device)

if __name__ == '__main__':

    start_text = input("Write any sentence here : \n")

    # the training should be here 
    train_losses, val_losses, tokens_seen = train_model_simple(gpt, train_loader, val_loader, optimizer, device, num_epochs = num_epochs, eval_freq = 5, eval_iter = 5, start_context = start_text, tokenizer = tokenizer)

    end_time = time.time()

    # total time spent on training
    execution_time_minutes = (end_time - start_time) / 60

    print(f"training completed in {execution_time_minutes:2f} minutes")

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(start_text, tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=40,
        temperature=0.8
    )


    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
