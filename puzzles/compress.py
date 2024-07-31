from tinygrad import Tensor
from arange import arange
from cumsum import cumsum


def compress(g: Tensor, v: Tensor, i: int) -> Tensor:
    if len(g.shape) > 1 or len(v.shape) > 1 or v.shape[0] != g.shape[0]:
        raise ValueError(f"g and v must be one-dimensional vectors of the same size")

    return (g * cumsum(1 * g) == (arange(i) + 1)[:, None]) @ v


if __name__ == "__main__":
    print(compress(Tensor([1, 0, 1]), Tensor([2, 4, 6]), 3).numpy())
