from tinygrad import Tensor


def where(q: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return q * a + q.logical_not() * b


if __name__ == "__main__":
    print(where(Tensor([False, True]), Tensor([1]), Tensor([3])).numpy())
