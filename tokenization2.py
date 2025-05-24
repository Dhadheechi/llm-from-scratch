#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)


# In[3]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])


# In[4]:


import re
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()] # removes whitespaces
print(result)


# In[5]:


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()] # removes whitespaces
print(len(preprocessed))


# In[6]:


print(preprocessed[:30])


# In[7]:


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size) # number of unique tokens


# In[8]:


vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break
# all the capital tokens (113) followed by the small letter tokens


# In[9]:


" ".join(["a", "b", "c", "."])


# In[10]:


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text): # take the text and encode it into token ids
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()] 
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids): # take the token ids and decode them back into text
        text = " ".join([self.int_to_str[i] for i in ids]) # all the tokens separated by spaces
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# In[11]:


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


# In[12]:


print(tokenizer.decode(ids))


# In[1]:


# text = "Hello, do you like the tea?"
# print(tokenizer.encode(text))


# In[14]:


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))


# In[15]:


for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# In[16]:


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab # to convert the tokens to token_IDs, we have to use a vocabulary
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text): # take the text and encode it into token ids
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()] 
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids): # take the token ids and decode them back into text
        text = " ".join([self.int_to_str[i] for i in ids]) # all the tokens separated by spaces
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # replaces spaces before punctuations
        return text


# In[17]:


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)


# In[18]:


tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)


# In[19]:


text = tokenizer.decode(ids)
print(text)


# In[ ]:


get_ipython().run_line_magic('pip', 'install tiktoken')


# In[21]:


from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))


# In[22]:


tokenizer = tiktoken.get_encoding("gpt2")


# In[23]:


text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)


# In[24]:


strings = tokenizer.decode(integers)
print(strings)


# In[25]:


word = "Akwirw ier"
ints = tokenizer.encode(word, allowed_special={"<|endoftext|>"})
print(ints)


# In[26]:


strs = tokenizer.decode(ints)
print(strs)


# In[27]:


for i in ints:
    print(tokenizer.decode([i])) # each argument, even if a single token, has to be passed as a list


# In[28]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text)) # number of tokens


# In[29]:


enc_sample = enc_text[50:] # remove the first 50 tokens
context_size = 4


# In[30]:


for i in range(1, context_size+1):
    sample = enc_sample[:i]
    desired = enc_sample[i]
    print(sample, "--->", desired)


# In[31]:


for i in range(1, context_size+1):
    sample = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(sample), "--->", tokenizer.decode([desired]))


# In[47]:


from torch.utils.data import Dataset, DataLoader

# A dataset for batched inputs and targets
class GPTDatasetV1(Dataset): # a subclass of the pytorch dataset
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+ max_length] # each row contains max_length token IDs
            target_chunk = token_ids[i+1: i+ max_length+1] # shifted by one unit forward

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.target_ids) # this len method and the following one make use of the attributes self.input, self.target

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    


# In[66]:


# A dataloader to generate batches with input-output pairs

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # the dataset is fed into the dataloader, which creates the batches
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers = num_workers
    )

    return dataloader


# In[55]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs: ", inputs)
print("targets: ", targets)


# In[57]:


second_batch = next(data_iter)
print(type(second_batch))


# In[58]:


vocab_size = 6
output_dim = 3


# In[59]:


torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)


# In[60]:


print(type(embedding_layer)) # we can perform a lookup for individual token IDs to get the embedding


# In[61]:


print(embedding_layer(torch.tensor([3])))


# In[64]:


input_ids = torch.tensor([2, 3, 5, 1])
print(embedding_layer(input_ids)) # we are calling the indices instead of indexing into them


# In[65]:


vocab_size = 50287
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# In[67]:


max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8,
                                  max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter) # to access the next batch
print("Inputs: ", inputs)
print("Input size: ", inputs.shape)


# In[68]:


token_embeddings = token_embedding_layer(inputs)


# In[70]:


print(token_embeddings.shape) # a 256 dimensional vector for each token ID (can be thought of as going into the plane, i.e., third dimension)


# In[71]:


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # perform the lookups 
print(pos_embeddings.shape)


# In[72]:


input_embedings = token_embeddings + pos_embeddings # adding (8, 4, 256) and (4, 256) via broadcasting
print(input_embedings.shape)


# In[ ]:




