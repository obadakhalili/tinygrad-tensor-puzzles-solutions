from tinygrad import Tensor
from cumsum import cumsum
from ones import ones


def linspace(i: int, j: int, k: int) -> Tensor:
    return ones(k) * i + (cumsum(ones(k) * 0 + (step := (j - i) / (k - 1))) - step)


if __name__ == "__main__":
    print(linspace(10, 1, 5).numpy())
