import torch

with open('../our.pt', "rb") as f:
    state = torch.load(f, map_location=torch.device("cpu"))

if "args" in state and state["args"] is not None:
    args = state["args"]