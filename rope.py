from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to Lecture 5 slides in https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 49 (linked above).

    # Referenced https://github.com/karpathy/llama2.c/blob/master/model.py
    # as mentioned in Piazza post
    # half_d = head_dim // 2
    pair_indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = theta ** (- pair_indices / head_dim)
    t = torch.arange(seqlen, device=device, dtype=torch.float32)  # Token positions
    freqs = torch.outer(t, freqs).float()  # (seqlen, d/2)
    # ^ each entry (t, i) is the rotation angle for token pos t & pair index i (t . w_i)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    freqs_cos = reshape_for_broadcast(freqs_cos, query_real)
    freqs_sin = reshape_for_broadcast(freqs_sin, query_real)

    query_out_r = query_real * freqs_cos - query_imag * freqs_sin
    # print(f"shape: {query_out_r.shape}")
    # (1, 2, 2, 2)
    query_out_i = query_imag * freqs_cos + query_real * freqs_sin
    key_out_r = key_real * freqs_cos - key_imag * freqs_sin
    key_out_i = key_imag * freqs_cos + key_real * freqs_sin

    # stack: (bs, seqlen, num_heads, d/2, 2)
    # last dim: 0 = real, 1 = imag
    # flatten(3) : from dim 3 onwards flatten everyth to 1 dim
    query_out = torch.stack([query_out_r, query_out_i], dim=-1).flatten(3)  # (bs, seqlen, num_heads, d)
    key_out = torch.stack([key_out_r, key_out_i], dim=-1).flatten(3)
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
