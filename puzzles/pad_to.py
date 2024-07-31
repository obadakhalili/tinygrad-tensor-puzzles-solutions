from tinygrad import Tensor
from arange import arange


def pad_to(a: Tensor, i: int) -> Tensor:
    if len(a.shape) > 1:
        raise ValueError(f"tensor must be a vector")
    return a @ (arange(a.shape[0])[:, None] == arange(i))


if __name__ == "__main__":
    print(pad_to(Tensor([1, 2, 3]), 4).numpy())
