from .unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class NModel(nn.Module):
    def __init__(self, template_mesh, im_width, im_height, device1, device2):
        super().__init__()

        n_blocks = 4
        out_width = im_width // (n_blocks *2)
        out_height = im_height // (n_blocks *2)

        verts = template_mesh[0]
        faces = template_mesh[1]

        verts = verts.to(device2)
        faces = faces.to(device2)

        self.aspect_ratio = im_width / im_height
        self.enc = UNet(
            in_channels=3,
            out_channels=2,
            activation='leaky',
            normalization='batch',
            n_blocks=n_blocks
        ).to(device1)

        self.texture = NTextureFlow().to(device2)
        self.vertices = Decoder( (verts, faces), dim_in=(2**(4+n_blocks)) * out_width * out_height).to(device2)

        self.device1 = device1
        self.device2 = device2

    def forward(self, inp):

        x, features = self.enc(inp.to(self.device1))        

        texture, flow = self.texture( inp.to(self.device2), x.permute(0, 2, 3, 1).to(self.device2) )
        verts = self.vertices( features[-1].view(features[-1].shape[0], -1).to(self.device2) )

        return verts.to(self.device1), texture.to(self.device1), flow.to(self.device1)

# NOT USED - MEMORY CONSTRAINTS
# class NLight(nn.Module):
#     def __init__(self, dim_in, w, h):
#         super().__init__()

#         dims = 1024
#         self.l0 = nn.Conv2d(dim_in, 1024, kernel_size=5, stride=1, padding=2)
#         self.l1 = ResNetLayer(1024, 1024)
#         self.l2 = ResNetLayer(1024, 1024)
#         self.l3 = nn.Conv2d(1024, 4, kernel_size=5, stride=1, padding=2)
#         self.fc = nn.Linear( 4*w*h , 9)

#     def forward(self, encoding):

#         encoding = self.l0(encoding)
#         encoding = self.l1(encoding)
#         encoding = self.l2(encoding)
#         encoding = self.l3(encoding)
#         encoding = F.tanh( self.fc(encoding.view( encoding.shape[0], -1 )) )

#         return encoding

class NTexture(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(512, mode='bilinear')
        self.up2 = nn.Upsample(512, mode='bilinear')
        self.conv1 = nn.Conv2d(2, 3, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, img, flow):

        flow = self.up(flow.permute(0, 3, 1, 2))
        flow = F.relu(self.bn1( self.conv1(flow) ), inplace=True)
        flow = F.relu(self.bn2( self.conv2(flow) ))
        flow = torch.tanh(self.up2(flow))

        return flow, flow

class NTextureFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(512, mode='bilinear')
        self.up2 = nn.Upsample(512, mode='bilinear')
        self.conv1 = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(2)

    def forward(self, img, flow):

        flow = self.up(flow.permute(0, 3, 1, 2))
        flow = F.relu(self.bn1( self.conv1(flow) ), inplace=True)
        flow = torch.tanh(self.bn2( self.conv2(flow) ))
        flow = self.up2(flow)

        return torch.nn.functional.grid_sample(img, flow.permute(0, 2, 3, 1)), flow

class Decoder(nn.Module):
    def __init__(self, template_mesh, dim_in=2048, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()

        self.vertices_base = template_mesh[0].detach().clone()
        self.faces = template_mesh[1].detach().clone()

        self.nv = self.vertices_base.shape[0]
        self.nf = self.faces.shape[0]
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5
        self.eps = 1e-5

        dim = 2048*2
        dim_hidden = [2*dim, dim]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5

        return vertices