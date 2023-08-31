import torch

paras = torch.load(r'D:\算法交易\Algorithmic-Trading\AlgorithmicStrategy\TWAP_VWAP\MODEL_SAVE\1.ocet',map_location=torch.device('cpu'))["model_state_dict"]
print(paras)