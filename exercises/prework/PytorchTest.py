import torch

print('Torch version: ' + torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
print('device name: ' + torch.cuda.get_device_name(0))


a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))

b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))

c = a + b
print('Tensor c = ' + str(c))

print(torch.rand(3,3).cuda())


# import torchvision
# print(torchvision.__version__)
