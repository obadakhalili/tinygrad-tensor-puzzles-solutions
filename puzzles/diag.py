from tinygrad import Tensor
from arange import arange
from ones import ones


def diag(t: Tensor) -> Tensor:
    if len(t.shape) != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("only works for 2-dim square tensors")
    return t * (arange(t.shape[0])[:, None] == arange(t.shape[0])) @ ones(t.shape[0])

    # alternative solution:
    # return t[idx := list(range(t.shape[0])), idx]


if __name__ == "__main__":
    print(diag(Tensor([[1, 2, 3], [4, 5, 6], [4, 5, 6]])).numpy())
