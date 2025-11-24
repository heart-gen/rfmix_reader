import math
import torch
import numpy as np
from typing import Tuple

__all__ = [
    "hmm_interpolate",
    "split_to_haplotypes",
]

def build_log_emissions_from_anchors(
    obs_post: np.ndarray | torch.Tensor, eps_anchor: float = 1e-3,
    device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build log-emission tensor from anchor posteriors.

    Parameters
    ----------
    obs_post : array-like, shape (B, L, K)
        Posterior ancestry probabilities from LAI (e.g. RFMix) at anchor
        positions. Use NaN where no anchor information is available.
        B = number of sequences (e.g. haplotypes or samples)
        L = number of loci (chr21 positions)
        K = number of ancestries.

    eps_anchor : float, default 1e-3
        Small amount of smoothing to avoid zero probabilities at anchors:
        emission = (1-eps)*p_anchor + eps/K.

    device : "cuda" or "cpu"
    dtype  : torch dtype, usually float32 or float64

    Returns
    -------
    log_emission : torch.Tensor, shape (B, L, K)
        Log P(obs_t | state=k) for each sequence & locus.
    """
    if not torch.is_tensor(obs_post):
        obs_post = torch.as_tensor(obs_post, device=device, dtype=dtype)
    else:
        obs_post = obs_post.to(device=device, dtype=dtype)

    B, L, K = obs_post.shape

    # mask where all ancestries are NaN
    nan_mask = torch.isnan(obs_post)                  # (B,L,K)
    has_anchor = ~nan_mask.all(dim=-1)                # (B,L)

    # Replace NaNs with 0 for now
    obs_filled = obs_post.clone()
    obs_filled[nan_mask] = 0.0

    # Normalize each (B,L, K) row to sum to 1 if there is any signal
    row_sums = obs_filled.sum(dim=-1, keepdim=True)   # (B,L,1)
    row_sums = torch.clamp(row_sums, min=1e-12)
    p = obs_filled / row_sums                         # (B,L,K)

    # For non-anchor positions, emissions should be uninformative / uniform
    uniform = torch.full_like(p, 1.0 / K)
    p = torch.where(has_anchor.unsqueeze(-1), p, uniform)

    # Smooth a bit to avoid exact zeros
    p = (1.0 - eps_anchor) * p + eps_anchor * (1.0 / K)

    # Log-transform
    p = torch.clamp(p, min=1e-12)
    log_emission = torch.log(p)

    return log_emission


def build_log_transitions(
    pos_bp: np.ndarray | torch.Tensor, K: int, recomb_rate: float = 1e-8,
    device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build per-step log transition matrices based on distances between loci.

    States are ancestries 0..K-1. For step t:

        p_stay(t)   = exp(-recomb_rate * delta_bp)
        p_switch(t) = 1 - p_stay(t)

        P(k | j) =
          p_stay(t)                    if j == k
          p_switch(t) / (K-1)          if j != k

    Parameters
    ----------
    pos_bp : array-like, shape (L,)
        Monotonic positions for the chromosome (bp or cM).

    K : int
        Number of ancestries (states).

    recomb_rate : float, default 1e-8
        Recombination rate per base (or per unit of pos if using cM).

    device : "cuda" or "cpu"
    dtype  : torch dtype

    Returns
    -------
    log_T : torch.Tensor, shape (L-1, K, K)
        log transition matrices between t-1 and t.
    """
    if not torch.is_tensor(pos_bp):
        pos = torch.as_tensor(pos_bp, device=device, dtype=dtype)
    else:
        pos = pos_bp.to(device=device, dtype=dtype)

    L = pos.shape[0]
    if L < 2:
        raise ValueError("Need at least 2 positions to build transitions.")

    delta = pos[1:] - pos[:-1]   # (L-1,)
    delta = torch.clamp(delta, min=0.0)

    p_stay = torch.exp(-recomb_rate * delta)  # (L-1,)
    p_switch = 1.0 - p_stay                   # (L-1,)

    # Build T[t,:,:] with vectorized operations
    T = (p_switch / max(K - 1, 1)).view(-1, 1, 1).expand(-1, K, K)  # (L-1,K,K)

    # Adjust diagonal to p_stay
    eye = torch.eye(K, device=device, dtype=dtype).view(1, K, K)    # (1,K,K)
    T = T + (p_stay - p_switch / max(K - 1, 1)).view(-1, 1, 1) * eye

    T = torch.clamp(T, min=1e-12, max=1.0)
    log_T = torch.log(T)

    return log_T


def _forward_backward(
    log_emission: torch.Tensor, log_T: torch.Tensor, log_pi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run forward-backward for a batch of sequences.

    Parameters
    ----------
    log_emission : (B, L, K) tensor
    log_T        : (L-1, K, K) tensor
    log_pi       : (K,) tensor

    Returns
    -------
    gamma      : (B, L, K) posterior P(state | all obs)
    log_alpha  : (B, L, K) forward log-probabilities
    log_beta   : (B, L, K) backward log-probabilities
    """
    device = log_emission.device
    B, L, K = log_emission.shape

    if log_T.shape != (L - 1, K, K):
        raise ValueError(f"log_T must be (L-1,K,K), got {log_T.shape}")

    log_alpha = torch.empty((B, L, K), device=device, dtype=log_emission.dtype)
    log_beta = torch.empty_like(log_alpha)

    # Initial
    log_alpha[:, 0, :] = log_pi.view(1, -1) + log_emission[:, 0, :]

    # Forward
    for t in range(1, L):
        a_prev = log_alpha[:, t - 1, :].unsqueeze(-1)
        trans_t = log_T[t - 1].unsqueeze(0)
        log_alpha[:, t, :] = torch.logsumexp(a_prev + trans_t, dim=1) + log_emission[:, t, :]

    # Backward
    log_beta[:, -1, :] = 0.0
    for t in range(L - 2, -1, -1):
        b_next = log_beta[:, t + 1, :] + log_emission[:, t + 1, :]
        b_next = b_next.unsqueeze(1)
        trans_t = log_T[t].unsqueeze(0)
        log_beta[:, t, :] = torch.logsumexp(trans_t + b_next, dim=-1)

    # Posterior gamma ~ alpha * beta
    log_gamma = log_alpha + log_beta
    log_norm = torch.logsumexp(log_gamma, dim=-1, keepdim=True)
    gamma = torch.exp(log_gamma - log_norm)

    return gamma, log_alpha, log_beta


def hmm_interpolate(
    pos_bp: np.ndarray | torch.Tensor, obs_post: np.ndarray | torch.Tensor,
    recomb_rate: float = 1e-8, eps_anchor: float = 1e-3,
    device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,
    batch_size_seqs: int = 128,
) -> np.ndarray:
    """
    High-level HMM interpolation for local ancestry.

    Parameters
    ----------
    pos_bp : (L,) array-like
        Positions for chr21 (bp or cM, monotonic).

    obs_post : (B, L, K) array-like
        Anchor posteriors from LAI for B sequences and K ancestries.
        Use NaN across all K at (b,t) to indicate "no anchor" at locus t
        for sequence b. Typically B is the number of haplotypes or samples.

    recomb_rate : float, default 1e-8
        Recombination rate per bp (or per unit of pos if using cM).

    eps_anchor : float, default 1e-3
        Smoothing for anchor emissions to avoid zeros.

    device : "cuda" or "cpu"
    dtype  : torch dtype
    batch_size_seqs : int, default 128
        Number of sequences (B dimension) to process at a time to
        control VRAM usage.

    Returns
    -------
    gamma_np : np.ndarray, shape (B, L, K)
        Posterior P(state=k | all anchors) at each locus.

    Notes
    -----
    - This function builds log-emissions and log-transitions once, then
      runs forward-backward in batches along the sequence dimension.
    - You can decode hard states via gamma_np.argmax(-1).
    """
    # Make sure obs_post is on CPU first to slice easily for batching
    if isinstance(obs_post, np.ndarray):
        obs_tensor = torch.as_tensor(obs_post, dtype=dtype)
    else:
        obs_tensor = obs_post.detach().to(dtype)

    B, L, K = obs_tensor.shape

    # Build shared transition matrices on device
    log_T = build_log_transitions(
        pos_bp=pos_bp, K=K, recomb_rate=recomb_rate,
        device=device, dtype=dtype,
    )

    # Initial distribution (uniform over ancestries)
    log_pi = torch.full((K,), -math.log(K), device=device, dtype=dtype)
    gamma_out = np.empty((B, L, K), dtype=np.float32)

    # Process sequences in mini-batches along B
    for b_start in range(0, B, batch_size_seqs):
        b_end = min(b_start + batch_size_seqs, B)
        obs_batch = obs_tensor[b_start:b_end].to(device)

        # Build log emissions for this batch
        log_emission = build_log_emissions_from_anchors(
            obs_post=obs_batch, eps_anchor=eps_anchor,
            device=device, dtype=dtype,
        )

        # Run forward-backward
        gamma, _, _ = _forward_backward(
            log_emission=log_emission, log_T=log_T, log_pi=log_pi,
        )
        gamma_out[b_start:b_end] = gamma.detach().cpu().numpy().astype(np.float32)
        torch.cuda.empty_cache()

    return gamma_out


def split_to_haplotypes(
    admix_summed: np.ndarray, hap_index: np.ndarray,
    missing_as_nan: bool = True,
) -> np.ndarray:
    """
    Convert individual-level summed ancestry (admix_summed) into
    haplotype-level anchor posteriors suitable for `hmm_interpolate`.

    Parameters
    ----------
    admix_summed : np.ndarray, shape (L, nsamples, npops)
        Summed ancestry counts/probabilities across the 2 haplotypes
        for each sample and ancestry.

    hap_index : np.ndarray, shape (nsamples, npops, 2)
        Mapping from (sample, ancestry, hap) -> original column index in X.
        Not strictly needed for building obs_post, but used to define
        the hap-ordering and to map results back later if desired.

    missing_as_nan : bool, default True
        If True, loci where total ancestry sum is 0 will be marked as
        missing (all NaNs) so that the HMM treats them as uninformative.

    Returns
    -------
    obs_post_haps : np.ndarray, shape (B, L, K)
        where B = 2 * nsamples, K = npops. For each sample j, its two
        haplotype sequences are at indices 2*j and 2*j+1. At locus t
        the probability vector over ancestries is proportional to the
        summed counts in admix_summed[t, j, :].
    """
    A = np.asarray(admix_summed, dtype=np.float32)      # (L, nsamples, npops)
    L, nsamples, npops = A.shape

    # Move sample to axis 0 for convenience: (nsamples, L, npops)
    A_samp = np.transpose(A, (1, 0, 2))

    # Sum over ancestries at each (sample, locus)
    row_sums = A_samp.sum(axis=-1, keepdims=True)      # (nsamples, L, 1)

    # Identify positions with no ancestry info (sum == 0)
    zero_mask = (row_sums == 0.0)

    # Avoid division by zero
    row_sums_safe = row_sums.copy()
    row_sums_safe[zero_mask] = 1.0

    # Convert counts to per-haplotype ancestry distribution
    P = A_samp / row_sums_safe

    if missing_as_nan:
        P[zero_mask.repeat(npops, axis=2)] = np.nan

    return P.repeat(2, axis=0)
