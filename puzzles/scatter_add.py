from tinygrad import Tensor
from arange import arange


def scatter_add(input: Tensor, index: Tensor, j: int) -> Tensor:
    if len(input.shape) > 1 or len(input.shape) != len(index.shape) or j < 0:
        raise ValueError(
            f"input and index must be 1D tensors of the same length, and j to be a positive integer"
        )
    return input @ (index[:, None] == arange(j))


if __name__ == "__main__":
    print(scatter_add(Tensor([1, 3, 4, 6, 3]), Tensor([3, 2, 1, 3, 0]), 5).numpy())
