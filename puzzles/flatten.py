from tinygrad import Tensor, dtypes
from arange import arange


def flatten(a: Tensor):
    if len(a.shape) != 2:
        raise ValueError("input must be 2D")

    def floor_div(t: Tensor, x: int):
        return (t / x).cast(dtypes.int)

    def mod(t: Tensor, x: int):
        return t - floor_div(t, x) * x

    return a[
        floor_div(arange(p := a.shape[0] * a.shape[1]), a.shape[1]),
        mod(arange(p), a.shape[1]),
    ]


if __name__ == "__main__":
    print(flatten(Tensor([[1, 2, 3], [4, 5, 6], [7, 7, 7]])).numpy())
