import os
from tinygrad import Tensor
from arange import arange

# default backend, CLANG, throws "error: self-comparison always evaluates to false"
os.environ["LLVM"] = "1"


def ones(len: int) -> Tensor:
    len_arange = arange(len)
    return (len_arange == len_arange) * 1


if __name__ == "__main__":
    print(ones(10).numpy())
