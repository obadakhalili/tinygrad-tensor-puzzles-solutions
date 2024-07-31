from tinygrad import Tensor
from ones import ones
from arange import arange


def bincount(a: Tensor, i: int) -> Tensor:
    if len(a.shape) > 1 or i < 0:
        raise ValueError(
            f"tensor must be a vector and i must be a non-negative integer"
        )
    return ones(a.shape[0]) @ (a[:, None] == arange(i))


if __name__ == "__main__":
    print(bincount(Tensor([1, 2, 1, 4, 2, 1, 6, 9]), 5).numpy())
