from tinygrad import Tensor
from arange import arange


def diff(v: Tensor) -> Tensor:
    if len(v.shape) > 1:
        raise ValueError("only works for 1-dim tensors")
    return v[arange(v.shape[0] - 1) + 1] - v[arange(v.shape[0] - 1)]


if __name__ == "__main__":
    print(diff(Tensor([1, 2, 4, 7, 0])).numpy())
