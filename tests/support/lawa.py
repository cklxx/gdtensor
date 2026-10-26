"""Shared helpers for LAWA (Latest Weight Averaging) tests."""


def normalize_weights(torch, weights):
    """Validate caller-supplied LAWA weights and normalize them to sum 1."""
    w = torch.tensor([float(x) for x in weights], dtype=torch.float64)
    if w.numel() == 0 or bool(torch.isnan(w).any()) or bool(torch.isinf(w).any()):
        raise ValueError("weights must be finite and non-empty")
    if bool((w < 0).any()):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return (w / total).tolist()
