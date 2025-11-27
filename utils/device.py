import os

import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)
        if device != 'None':
            self.gpu_list = [i for i in range(len(device.split(',')))]
            # print(self.gpu_list)
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            self.device = torch.device(f"cuda:{self.gpu_list[0]}" if len(self.gpu_list) > 0 else "cpu")
            self.occupy_gpu(self.gpu_list)
        else:
            self.device = torch.device("cpu")
        self.output_device = self.device

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        """
        递归地将数据移到指定设备上。
        支持的类型：Tensor、list、tuple、dict、str、int
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self.data_to_device(d) for d in data]
        elif isinstance(data, tuple):
            return tuple(self.data_to_device(d) for d in data)
        elif isinstance(data, dict):
            return {key: self.data_to_device(value) for key, value in data.items()}
        elif isinstance(data, str):  # 处理字符串类型
            return data
        elif isinstance(data, int):  # 处理整数类型
            return data
        else:
            raise ValueError(f"Unknown data type: {type(data)}")

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
