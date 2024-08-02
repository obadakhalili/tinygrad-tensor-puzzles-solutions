from tinygrad import Tensor
from triu import triu
from where import where
from ones import ones
from arange import arange


def bucketize(v: Tensor, b: Tensor) -> Tensor:
    if len(v.shape) > 1 or len(v.shape) != len(b.shape):
        raise ValueError(f"input and index must be 1D tensors of the same length")

    return (v[:, None] >= b) @ ones(b.shape[0])


if __name__ == "__main__":
    print(bucketize(Tensor([2, 3, -1, 1]), Tensor([0, 2, 3])).numpy())
