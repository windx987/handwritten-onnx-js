import torch
from template import Net

def main(path):
  pytorch_model = Net()
  pytorch_model.load_state_dict(torch.load(path))
  pytorch_model.eval()
  dummy_input = torch.zeros(280 * 280 * 4)
  torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)


if __name__ == '__main__':
  path = '/home/teerawat.c/projects/handwritten-onnx-js/models/converter-models/pytorch_model.pt'
  main(path)