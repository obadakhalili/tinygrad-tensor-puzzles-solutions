from tinygrad import Tensor


def arange(i: int) -> Tensor:
    return Tensor.arange(i)


if __name__ == "__main__":
    print(arange(5).numpy())
