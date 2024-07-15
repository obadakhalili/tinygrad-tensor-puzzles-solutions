from tinygrad import Tensor


def flip(v: Tensor, i: int) -> Tensor:
    if len(v.shape) > 1:
        raise ValueError(f"tensor must be a vector")

    return v[:i:][::-1]


if __name__ == "__main__":
    print(flip(Tensor([1, 2, 3, 4, 5]), 4).numpy())
