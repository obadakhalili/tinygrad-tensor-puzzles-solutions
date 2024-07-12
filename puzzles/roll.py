from tinygrad import Tensor
from arange import arange


def roll(v: Tensor, i: int) -> Tensor:
    if len(v.shape) > 1 or i < 1 or i >= v.shape[0]:
        raise ValueError(
            f"tensor must be a vector and shift must be in {(1, v.shape[0] - 1)}"
        )

    return v[list(range(v.shape[0]))[-i:] + list(range(v.shape[0]))[:-i]]

    # alternative solution
    # return v[arange(v.shape[0]) - i]


if __name__ == "__main__":
    print(roll(Tensor([1, 2, 3]), 3).numpy())
