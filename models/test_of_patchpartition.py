import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载图像并进行预处理
# img = Image.open('/data/ML_document/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG')
img = Image.open('D:\\Work\\datasets\\coco128\\images\\train2017\\000000000009.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
x = transform(img).unsqueeze(0)
# print(x.shape)
# x = x.permute(0, 2, 3, 1)
# # plt.matshow(x.squeeze(0).squeeze(-1))
# x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
# x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
# x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
# x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
# print(x0.shape)
# x0 = x0.squeeze(0).squeeze(-1)
# x0_0 = x0.squeeze(0)[:,:,0]
# x0_1 = x0.squeeze(0)[:,:,1]
# x0_2 = x0.squeeze(0)[:,:,2]
# print(x0.shape)
# print(x0_1.shape)
# plt.matshow(x0)
# plt.matshow(x0_0)
# plt.matshow(x0_1)
# plt.matshow(x0_2)
# plt.show()


# # 使用上述代码段进行处理
# class SwinTransformerBlock(torch.nn.Module):
#     def __init__(self, input_resolution, num_heads, window_size, shift_size, mlp_ratio=4.0):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio

#         self.norm = torch.nn.LayerNorm(input_resolution * input_resolution * 3)
#         self.reduction = torch.nn.Linear(input_resolution * input_resolution * 12, input_resolution * input_resolution * 3)

#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

#         x = x.view(B, H, W, C)

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

#         return x

# block = SwinTransformerBlock(input_resolution=112, num_heads=8, window_size=7, shift_size=0)
# y = block(x)

# 输出处理后的张量
# print(y.shape)


# test the "build layers" of the SwinTransformer in Line547-Line563
drop_path_rate = 0.1
depths=[2, 2, 6, 2]
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
for i_layer in range(len(depths)):
    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
    print(drop_path)