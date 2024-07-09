import os
from tinygrad import Tensor
from arange import arange
from where import where
from ones import ones


def diag(t: Tensor) -> Tensor:
    if len(t.shape) != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("only works for 2-dim square tensors")
    height, width = t.shape
    height_arange = arange(height)
    keep_map = height_arange[:, None] == height_arange
    diag_mat = where(keep_map, t, Tensor([0])) # or t * keept_map
    return diag_mat @ ones(height)

    # alternative solution:
    # height = t.shape[0]
    # idx = arange(height)
    # diag = t[idx, idx]
    # return diag


if __name__ == "__main__":
    print(diag(Tensor([[1, 2, 3], [4, 5, 6], [4, 5, 6]])).numpy())
