from tinygrad import Tensor
from arange import arange
from triu import triu


def cumsum(t: Tensor) -> Tensor:
    if len(t.shape) > 1:
        raise ValueError("only works for 1-dim tensors")
    # NOTE: why it doesn't work if the operand on the left was indexed?
    return (arange(t.shape[0]) <= arange(t.shape[0])[:, None]) @ t

    # alternative solution:
    # return t @ triu(t.shape[0])


if __name__ == "__main__":
    print(cumsum(Tensor([1, 2, 3])).numpy())
