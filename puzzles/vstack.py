from tinygrad import Tensor


def vstack(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) != len(b.shape) and len(a.shape) == 1:
        raise ValueError("only works for 1-dim tensors")
    return Tensor([[1], [0]]) * a + Tensor([[0], [1]]) * b


if __name__ == "__main__":
    print(vstack(Tensor([1, 2, 3]), Tensor([4, 5, 6])).numpy())
