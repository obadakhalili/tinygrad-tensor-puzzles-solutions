from tinygrad import Tensor


def outer(a: Tensor, b: Tensor) -> Tensor:
    return a[:, None] * b


if __name__ == "__main__":
    print(outer(Tensor([1, 2]), Tensor([1, 2, 3, 4, 5])).numpy())
