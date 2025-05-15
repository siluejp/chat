import torch
print(torch.backends.mps.is_available())

tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)

floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)

tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])

print(tensor2d)
