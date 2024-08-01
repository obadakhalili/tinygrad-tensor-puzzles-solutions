from tinygrad import Tensor
from ones import ones


def repeat(a: Tensor, i: int) -> Tensor:
    if len(a.shape) > 1 or i < 0:
        raise ValueError(
            f"tensor must be a vector and i must be a non-negative integer"
        )
    return a + (ones(i) * 0)[:, None]


if __name__ == "__main__":
    print(repeat(Tensor([1, 2, 3]), 2).numpy())
