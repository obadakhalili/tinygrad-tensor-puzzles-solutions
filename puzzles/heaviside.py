from tinygrad import Tensor


def heaviside(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) > 1 or len(a.shape) != len(b.shape):
        raise ValueError(
            f"a and b must be 1D tensors of the same length, and j to be a positive integer"
        )
    return (a > 0) + (a == 0) * b


if __name__ == "__main__":
    print(heaviside(Tensor([1, 0, 3, -1, 0]), Tensor([3, 2, 1, 3, 5])).numpy())
