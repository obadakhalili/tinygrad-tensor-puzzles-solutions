from tinygrad import Tensor
from ones import ones


def sum(t: Tensor) -> Tensor:
    if len(t.shape) > 1:
        raise ValueError("only works for 1-dim tensors")
    return t @ ones(t.shape[0])


if __name__ == "__main__":
    print(sum(Tensor([1, 2, 3])).numpy())
