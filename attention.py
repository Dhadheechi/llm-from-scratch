#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your 
[0.55, 0.87, 0.66], # journey 
[0.57, 0.85, 0.64], # starts 
[0.22, 0.58, 0.33], # with 
[0.77, 0.25, 0.10], # one 
[0.05, 0.80, 0.55]] # step 
) # we have six input tokens


# In[3]:


query = inputs[1] # the second input token serves as the query
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(query, inputs[i]) # dot product of the query with the ith token
print(attention_scores_2)


# As expected, the dot product of the query with itself has the highest value, i.e., attention score. The query attends the most to itself, as expected intuitively. Hence, the context vector z2 also gives the highest attention weight to its own corresponding input vector x2

# In[4]:


attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
print("Attention weights:", attention_weights_2)
print("Sum:", attention_weights_2.sum())


# In[5]:


query = inputs[1]
context_vec_2 = torch.zeros(query.shape) # context vector for the second input token
for idx, x_i in enumerate(inputs):
    context_vec_2 += attention_weights_2[idx] * x_i

print(context_vec_2) # an enhanced representation of the second input token


# In[6]:


attention_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i,j] = torch.dot(x_i, x_j)
print(attention_scores)


# In[7]:


attention_scores = inputs @ inputs.T
print(attention_scores)


# In[8]:


attention_weights = torch.softmax(attention_scores, dim=-1) # along the last dimension, i.e., dimension 1


# In[9]:


print(torch.sum(attention_weights, dim=1))


# In[10]:


context_vecs = attention_weights @ inputs # softmax(XX^T)X as per bishop. Just a matrix multiplication


# In[11]:


context_vecs


# Now we implement self-attention, with all the query, key and value matrices

# We again start with the goal of calculating the context vector of a single input token, before generalizing it.

# In[12]:


x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2


# In[13]:


x_2.shape


# In[14]:


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


# In[15]:


query_2 = x_2 @ W_query # x_2 is automatically broadcasted as a row vector [1, 3] to support the matrix multiplication 
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)


# In[16]:


W_query.shape


# In[17]:


torch.ones(3) @ torch.ones(3,2)


# In[18]:


torch.ones(3,2) @ torch.ones(2)


# In[19]:


keys = inputs @ W_key # (N, 3) @ (3,2) = (N,2)
values = inputs @ W_value
print("keys.shape", keys.shape)
print("values.shape", values.shape)


# In[20]:


attention_scores_2 = query_2 @ keys.T # all the attention scores for a given query; a bunch of dot products
print(attention_scores_2)


# In[21]:


d_k = keys.shape[-1] # the embedding dimension of the keys
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)
print(attention_weights_2)


# In[22]:


context_vec2 = attention_weights_2 @ values # a row vector times a matrix = linear combination of the rows of the matrix


# In[23]:


# A compact self-attention class
import torch.nn as nn
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__() # intializing the parent class
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) # a (3,2) matrix
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, X): # takes the input data matrix and calculates the output matrix of context vectors
        queries = X @ self.W_query
        keys = X @ self.W_key
        values = X @ self.W_value

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=1)
        context_vecs = attention_weights @ values
        return context_vecs



# In[24]:


torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
print(sa_v1(inputs)) # I think calling the class itself is equivalent to calling the forward method, in torch.nn.Module


# In[25]:


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__() # intializing the parent class
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, X): 
        queries = self.W_query(X) # nn.Linear stores the weight matrices in transposed form, i.e., it's a (2,3) matrix
        keys = self.W_key(X) # we do X @ (what's stored in nn.Linear(W.T)).T, esentially X@W
        values = self.W_value(X)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=1)
        context_vecs = attention_weights @ values
        return context_vecs


# In[26]:


torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
print(sa_v2(inputs))


# In[27]:


sa_v1.W_query.data = sa_v2.W_query.weight.T
sa_v1.W_key.data = sa_v2.W_key.weight.T
sa_v1.W_value.data = sa_v2.W_value.weight.T


# In[28]:


print(sa_v1(inputs)) # verifying that the implementation is the same


# In[29]:


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)


# In[30]:


attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=1)
print(attention_weights)


# In[31]:


context_length = attention_weights.shape[0]
mask = torch.tril(torch.ones(context_length, context_length)) # returns the lower triangular part of the matrix
masked_weights = attention_weights * mask
print(masked_weights)


# In[32]:


row_sums = masked_weights.sum(dim=-1, keepdim=True) # has shape (context_length, 1) instead of (context_length,)
masked_weights_norm = masked_weights / row_sums
print(masked_weights_norm)


# In[33]:


mask_inf = torch.triu(torch.ones(context_length, context_length), diagonal=1) # diagonal=1 => exclude the main diagonal
print(mask_inf)


# In[34]:


masked_attention_scores = attention_scores.masked_fill(mask_inf.bool(), -torch.inf)


# In[35]:


attention_weights = torch.softmax(masked_attention_scores/keys.shape[-1]**0.5, dim=-1)
print(attention_weights)


# In[37]:


dropout = torch.nn.Dropout(0.5)


# In[38]:


type(dropout)


# In[39]:


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)


# In[42]:


valid = ~mask_inf.bool()


# In[43]:


torch.bernoulli(valid.float() * 0.5)


# In[46]:


class CausalAttention(nn.Module): # also handles a batch of input tokens, i.e., a three dimensional tensor
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )
    
    def forward(self, X):
        batch_size, num_tokens, d_in = X.shape
        queries = self.W_query(X)
        keys = self.W_key(X)
        values = self.W_value(X)

        attention_scores = queries @ keys.transpose(1,2) # i.e., transpose only the second and third dimensions
        attention_scores = attention_scores / keys.shape[-1]**0.5

        # Get the causal mask (1s in upper triangular part)
        causal_mask = self.mask.bool()[:num_tokens, :num_tokens]
        
        # Always apply causal mask first
        attention_scores.masked_fill_(causal_mask, -torch.inf)

        if self.training and self.dropout.p > 0:
            valid_positions = ~causal_mask # inverting the causal mask

            # create a random dropout mask
            dropout_mask = torch.bernoulli(valid_positions.float() * self.dropout.p)

            attention_scores = attention_scores.masked_fill(dropout_mask.bool(), -torch.inf)

        attention_weights = torch.softmax(attention_scores, dim=-1) # over the last dimension, i.e., the token embedding dimension
        context_vecs = attention_weights @ values
        return context_vecs



# In[47]:


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0)
context_vecs = ca(batch)
print(context_vecs.shape)


# In[48]:


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout) for _ in range(num_heads)]
        )

    def forward(self, X):
        return torch.cat([head(X) for head in self.heads], dim=-1) # concatenate along the last dimension, i.e., embedding dimension


# In[51]:


torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0, num_heads=2)
context_vecs = mha(batch) # applying the forward method

print(context_vecs)
print("Context vectors shape: ", context_vecs.shape)


# In[72]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): 
        # here d_out = num_heads * output dim of a single head
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # output dimension of a single head
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, X):
        b, num_tokens, d_in = X.shape # here num_tokens need not be context_length (which is the maximum number of input tokens our model can handle)
        queries = self.W_query(X)
        keys = self.W_key(X)
        values = self.W_value(X) # Q, K, V has shape (b, num_tokens, d_out)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1,2) # Q, K, V have shape (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1,2)  
        values = values.transpose(1,2)

        attention_scores = queries @ keys.transpose(2, 3) # the scores has shape (b, num_heads, num_tokens, num_tokens)
        # start: this part doesn't change
        attention_scores = attention_scores / keys.shape[-1]**0.5

        # Get the causal mask (1s in upper triangular part)
        causal_mask = self.mask.bool()[:num_tokens, :num_tokens]
        
        # Always apply causal mask first
        attention_scores.masked_fill_(causal_mask, -torch.inf)

        if self.training and self.dropout.p > 0:
            valid_positions = ~causal_mask # inverting the causal mask

            # create a random dropout mask
            dropout_mask = torch.bernoulli(valid_positions.float() * self.dropout.p)

            attention_scores = attention_scores.masked_fill(dropout_mask.bool(), -torch.inf)

        attention_weights = torch.softmax(attention_scores, dim=-1) 
        # End of the part that doesn't change

        context_vecs = attention_weights @ values # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
         # = (b, num_heads, num_tokens, head_dim --> we got the context vectors for all the attention heads simultaneously 

        # now we reshape context vectors to the shap (b, num_of_tokens, d_out), i.e., flatten it; combine the heads
        context_vecs = context_vecs.transpose(1,2)
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)
        context_vecs = self.output_proj(context_vecs)

        return context_vecs


        


# In[73]:


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 4
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[74]:


torch.manual_seed(123)
batch_size, context_length, d_in = 1, 1024, 768
batch = torch.rand(1024, 768)
batch = batch.reshape(1, 1024, 768)
d_out = 768*12
mha_gpt2 = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vecs = mha_gpt2(batch)
print("context_vecs.shape:", context_vecs.shape)


# In[67]:


a = torch.rand(5,5)
b = a.reshape(1, -1)
b.shape


# In[ ]:




