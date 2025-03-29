import torch
import subprocess


def export_to_netron(model, dummy_input, filename="model.onnx"):
    torch.onnx.export(model, dummy_input, filename, 
                      input_names=["input"], output_names=["output"], opset_version=11)
    subprocess.run(["netron", filename])
