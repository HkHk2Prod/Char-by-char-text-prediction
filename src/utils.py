import torch

def detach_state(state: torch.Tensor | tuple | None) -> torch.Tensor | tuple | None:
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()