from tinygrad import Tensor
from arange import arange


def triu(x: int) -> Tensor:
    return (arange(x)[:, None] <= arange(x)) * 1


if __name__ == "__main__":
    print(triu(5).numpy())
