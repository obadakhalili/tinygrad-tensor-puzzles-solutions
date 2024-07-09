from tinygrad import Tensor
from arange import arange


def ones(len: int) -> Tensor:
    # return Tensor([1])._broadcast_to((len,))
    return (arange(len) >= 0) * 1


if __name__ == "__main__":
    print(ones(10).numpy())
