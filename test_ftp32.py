import torch
import numpy as np

x = torch.tensor([1000.0, 999.0, 995.0], dtype=torch.float16)
softmax_native = torch.exp(x) / torch.exp(x).sum()
softmax_stable = torch.softmax(x, dim=0)
print("native :", softmax_native)
print("stable :", softmax_stable)

arr = np.array([1e8, 1.0, -1e8], dtype=np.float32)

naive = arr.sum(dtype=np.float32)

def kahan_sum(a: np.ndarray) -> np.float32:
    s = np.float32(0.0); c = np.float32(0.0)
    for y in a:
        y = np.float32(y) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

print("naive:", naive, "kahan:", kahan_sum(arr))
