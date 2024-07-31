from tinygrad import Tensor
from arange import arange


def sequence_mask(values: Tensor, lengths: Tensor) -> Tensor:
    if (
        len(values.shape) != 2
        or len(lengths.shape) != 1
        or values.shape[0] != lengths.shape[0]
    ):
        raise ValueError(
            f"values must be 2D tensor and lengths must be 1D tensor with the same length as the first dimension of values"
        )
    return ((arange(values.shape[1]) + 1) <= lengths[:, None]) * values


if __name__ == "__main__":
    print(
        sequence_mask(
            Tensor([[1, 2, 3], [3, 1, 4], [1, 0, 3], [2, 3, 4], [3, 0, 0]]),
            Tensor([1, 2, 0, 3, 0]),
        ).numpy()
    )
