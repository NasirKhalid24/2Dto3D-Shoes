import os
import wandb
import torch
import lpips
import torchvision
import numpy as np
import kaolin as kal

from tqdm import tqdm

from modules.dataset import Shoes
from modules.model import NModel

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

batch_size=8
texture_res=512
aspect_ratio = 1000 / 750
im_height = 128 
im_width = int(im_height // aspect_ratio)
lr = 1e-4
scheduler_step_size = 500
scheduler_gamma = 0.5
num_epoch = 500
laplacian_weight = 0.1
flow_weight = 0.1
perceptual_weight = 0.01
flat_weight = 0.001
image_weight = 0.1
mask_weight = 0.9
views = 35


wandb.init(
    project="Shoe-Model",
    name="Default-Training",
    entity='nasirk24',
)
# GPU 
device = torch.device("cuda:0")
device1 = torch.device("cuda:1")
torch.cuda.set_device(device)

# Data
dataset = Shoes('./data')
dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=12)

# Initial Shape and Texture
mesh = kal.io.obj.import_mesh('./shapes/original/ready.obj', with_materials=True)

# Loss setup
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

vertices_init = mesh.vertices
N = vertices_init.shape[0]
center = vertices_init.mean(0)
scale = max((vertices_init - center).abs().max(0)[0])
vertices_init = vertices_init - center
vertices = vertices_init * (0.5 / float(scale))

vertices_init = vertices_init.detach()
faces = mesh.faces
uvs = mesh.uvs.unsqueeze(0)
face_uvs_idx = mesh.face_uvs_idx
face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()

nb_faces = faces.shape[0]
nb_vertices = mesh.vertices.shape[0]
face_size = 3

# USE THIS IF TENSORS SAVED ALREADY ELSE ABOVE - ABOVE IF FIRST TIME ALSO
# vertices_init = torch.load('./tensors/vertices.pt').to(device)
# faces = torch.load('./tensors/faces.pt').to(device)

# uvs = torch.load('./tensors/uvs.pt')
# face_uvs = torch.load('./tensors/faces_uvs.pt')
# face_uvs_idx = torch.load('./tensors/faces_uvs_idx.pt').to(device)

nb_faces = faces.shape[0]
nb_vertices = mesh.vertices.shape[0]
face_size = 3

# Model
model = NModel( (vertices_init, faces), im_width, im_height, device, device1)

# If finetuning
# model.load_state_dict(torch.load('model_400.pth'))
# model.eval()

faces = faces.to(device)
vertices_init = vertices_init.to(device)

## Set up auxiliary connectivity matrix of edges to faces indexes for the flat loss
edges = torch.cat([faces[:,i:i+2] for i in range(face_size - 1)] +
                  [faces[:,[-1,0]]], dim=0)

edges = torch.sort(edges, dim=1)[0]
face_ids = torch.arange(nb_faces, device=device, dtype=torch.long).repeat(face_size)
edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
nb_edges = edges.shape[0]
# edge to faces
sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
sorted_faces_ids = face_ids[order_edges_ids]
# indices of first occurences of each key
idx_first = torch.where(
    torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                            (1,0), value=1))[0]
nb_faces_per_edge = idx_first[1:] - idx_first[:-1]
# compute sub_idx (2nd axis indices to store the faces)
offsets = torch.zeros(sorted_edges_ids.shape[0], device=device, dtype=torch.long)
offsets[idx_first[1:]] = nb_faces_per_edge
sub_idx = (torch.arange(sorted_edges_ids.shape[0], device=device, dtype=torch.long) -
           torch.cumsum(offsets, dim=0))
nb_faces_per_edge = torch.cat([nb_faces_per_edge,
                               sorted_edges_ids.shape[0] - idx_first[-1:]],
                              dim=0)
max_sub_idx = 2
edge2faces = torch.zeros((nb_edges, max_sub_idx), device=device, dtype=torch.long)
edge2faces[sorted_edges_ids, sub_idx] = sorted_faces_ids

## Set up auxiliary laplacian matrix for the laplacian loss
vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(
    nb_vertices, faces)

# Optimizer
optim  = torch.optim.Adam(params=model.parameters(),
                          lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size,
                                            gamma=scheduler_gamma)


step_ = 0
t_loop = tqdm(range(num_epoch))

torch.save(model.state_dict(), './model_init.pth')

for epoch in t_loop:
    epoch_loss = 0
    for idx, data in enumerate(tqdm(dataloader, leave=False)):
        optim.zero_grad()
        
        gt_image = data['images'].to(device)
        gt_mask = data['masks'].to(device)
        cam_rot = data["R"].to(device)
        camera_trans = data["T"].to(device) 
        cam_proj = data['P'].to(device)
        
        vertices, texture, flow = model( gt_image[:, 0, ...] ) 
        
        # Flow Loss
        flow_loss = 1 - torch.nn.functional.grid_sample(gt_mask[:, 0, ...], flow.permute(0, 2, 3, 1)).mean()

        # Render Views
        view_idx = list(np.random.permutation( views ))[0]
        # view_idx = 0

        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                vertices,
                faces, cam_proj[0], camera_rot=cam_rot[:, view_idx, ...], camera_trans=camera_trans[:, view_idx, ...] # VIEW view_idx SELECTED
            )

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        face_attributes = [
            face_uvs.repeat(batch_size, 1, 1, 1).to(device),
            torch.ones((batch_size, nb_faces, 3, 1), device=device)
        ]

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            gt_image.shape[3], gt_image.shape[4], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        image = kal.render.mesh.texture_mapping(texture_coords,
                                                texture, 
                                                mode='bilinear')
        image = torch.clamp(image * mask, 0., 1.)
        
        ### Compute Losses ###
        image_loss = torch.mean(torch.abs(image - gt_image[:, view_idx, ...].permute(0, 2, 3, 1) ))
        mask_loss = kal.metrics.render.mask_iou(soft_mask,
                                                gt_mask[:, view_idx, ...].squeeze(1) )
        # laplacian loss
        vertices_mov = vertices - vertices_init
        vertices_mov_laplacian = torch.matmul(vertices_laplacian_matrix, vertices_mov)
        laplacian_loss = torch.mean(vertices_mov_laplacian ** 2) * nb_vertices * 3
        # flat loss
        mesh_normals_e1 = face_normals[:, edge2faces[:, 0]]
        mesh_normals_e2 = face_normals[:, edge2faces[:, 1]]
        faces_cos = torch.sum(mesh_normals_e1 * mesh_normals_e2, dim=2)
        flat_loss = torch.mean((faces_cos - 1) ** 2) * edge2faces.shape[0]

        # perceptual loss
        percep_loss = loss_fn_alex(  image.permute(0, 3, 1, 2) , gt_image[:, view_idx, ...], normalize=True ).mean()

        loss = (
            perceptual_weight * percep_loss +
            flow_loss * flow_weight +
            image_loss * image_weight +
            mask_loss * mask_weight +
            laplacian_loss * laplacian_weight +
            flat_loss * flat_weight
        )
        epoch_loss += loss
        loss.backward()

        

        wandb.log({
            "Total Loss": loss.item(),
            "Image Loss": (image_loss * image_weight).item(),
            "Flow loss": (flow_loss * flow_weight).item(),
            "Perceptual Loss": percep_loss.item(),
            "Mask Loss": (mask_loss * mask_weight).item(),
            "Laplacian Loss": (laplacian_loss * laplacian_weight).item(),
            "Flat Loss": (flat_loss * flat_weight).item(),
            "Learning Rate": optim.param_groups[0]['lr'],
        }, step=step_)
        step_ += 1

        ### Update the mesh ###
        optim.step()

        t_loop.set_description("Loss = %.6f" % loss)

        if idx % 100 == 0:
            torchvision.utils.save_image(image.permute(0, 3, 1, 2), './output/' + str(epoch) + str(idx) + "_render.jpeg")
            torchvision.utils.save_image(gt_image[:, view_idx, ...], './output/' + str(epoch) + str(idx) + "_target.jpeg")

            wandb.log({
                "Render": wandb.Image( './output/' + str(epoch) + str(idx) + "_render.jpeg" ),
                "Target": wandb.Image( './output/' + str(epoch) + str(idx) + "_target.jpeg" ),
            })

    epoch_loss = epoch_loss / (epoch+1)

    wandb.log({"Epoch Loss": epoch_loss, "Learning Rate": optim.param_groups[0]['lr'], "epoch": epoch})

   
    
    if epoch % 100 == 0 and epoch != 0:
        torch.save(model.state_dict(), './model_' + str(epoch) + '.pth')

    scheduler.step()
    print(f"Epoch {epoch} - loss: {float(loss)}")

torch.save(model.state_dict(), './model_done.pth')