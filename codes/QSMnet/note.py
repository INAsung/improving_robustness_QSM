import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("../save/qsmnet111/model/ep015.pth", map_location=device)
print(model)