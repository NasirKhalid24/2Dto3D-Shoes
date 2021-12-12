import os
import sys
import io
import uuid
import shutil
import tempfile
import zipfile
import base64

# API Imports
from flask import Flask, jsonify, request, make_response, Response, send_file, after_this_request


MOUNT_PATH = "./"
sys.path.append(MOUNT_PATH + "lib")

#################### UNET ####################
##############################################

import torch
from torch import nn
import torch.nn.functional as F

import torchvision.transforms as t

from PIL import Image

######################################################
######################################################

#################### UTILITIES #######################
######################################################

aspect_ratio = 1000 / 750
im_height = 128 
im_width = int(im_height // aspect_ratio)

tf = t.Compose(
    [
        t.Resize([im_width, im_height]),
        t.ToTensor()
    
    ]
)

def position_shoe(raw):
    im_r = raw.resize( (im_height, im_width))
    n = np.asarray( im_r )
    last_row = -1
    for row_num, row in enumerate(n):
        max_v = 255*3*row.shape[0]
        percent = sum(row.reshape(-1)) / max_v

        if percent < 0.95:
            last_row = row_num

    if last_row > 85:
        return im_r
    else:
        translate_by = 85 - last_row

        img_rt = im_r.transform(im_r.size, Image.AFFINE, (1, 0, 0, 0, 1, -translate_by), fillcolor=(255, 255, 255))
        return img_rt

def prepare_image(image_bytes):
    img = Image.open( image_bytes ).convert("RGB")
    img = position_shoe(img)

    return  tf( img )



"""
    PYTORCH3D
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""This module implements utility functions for loading and saving meshes."""
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

PathOrStr = Union[Path, str]

def save_obj(
    f,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    *,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    texture_map: Optional[torch.Tensor] = None,
) -> None:
    """
    Save a mesh to an .obj file.

    Args:
        f: File (str or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
        path_manager: Optional PathManager for interpreting f if
            it is a str.
        verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinate per vertex.
        faces_uvs: LongTensor of shape (F, 3) giving the index into verts_uvs for
            each vertex in the face.
        texture_map: FloatTensor of shape (H, W, 3) representing the texture map
            for the mesh which will be saved as an image. The values are expected
            to be in the range [0, 1],
    """
    if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
        message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
        message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
        raise ValueError(message)

    if texture_map is not None and (texture_map.dim() != 3 or texture_map.size(2) != 3):
        message = "'texture_map' should either be empty or of shape (H, W, 3)."
        raise ValueError(message)

    save_texture = all([t is not None for t in [faces_uvs, verts_uvs, texture_map]])
    output_path = Path(f)

    # Save the .obj file
    with open(f, "w") as f:
        if save_texture:
            # Add the header required for the texture info to be loaded correctly
            obj_header = "\nmtllib {0}.mtl\nusemtl mesh\n\n".format(output_path.stem)
            f.write(obj_header)
        _save(
            f,
            verts,
            faces,
            decimal_places,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            save_texture=save_texture,
        )

    # Save the .mtl and .png files associated with the texture
    if save_texture:
        image_path = output_path.with_suffix(".png")
        mtl_path = output_path.with_suffix(".mtl")
        if isinstance(f, str):
            # Back to str for iopath interpretation.
            image_path = str(image_path)
            mtl_path = str(mtl_path)

        # Save texture map to output folder
        # pyre-fixme[16] # undefined attribute cpu
        texture_map = texture_map.detach().cpu() * 255.0
        image = Image.fromarray(texture_map.numpy().astype(np.uint8))
        with open(image_path, "wb") as im_f:
            # pyre-fixme[6] # incompatible parameter type
            image.save(im_f)

        # Create .mtl file with the material name and texture map filename
        # TODO: enable material properties to also be saved.
        with open(mtl_path, "w") as f_mtl:
            lines = f"newmtl mesh\n" f"map_Kd {output_path.stem}.png\n"
            f_mtl.write(lines)


# TODO (nikhilar) Speed up this function.
def _save(
    f,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    *,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    save_texture: bool = False,
) -> None:

    if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    verts, faces = verts.cpu(), faces.cpu()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    if save_texture:
        if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
            message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)

        if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
            message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
            raise ValueError(message)

        # pyre-fixme[16] # undefined attribute cpu
        verts_uvs, faces_uvs = verts_uvs.cpu(), faces_uvs.cpu()

        # Save verts uvs after verts
        if len(verts_uvs):
            uV, uD = verts_uvs.shape
            for i in range(uV):
                uv = [float_str % verts_uvs[i, j] for j in range(uD)]
                lines += "vt %s\n" % " ".join(uv)

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            if save_texture:
                # Format faces as {verts_idx}/{verts_uvs_idx}
                face = [
                    "%d/%d" % (faces[i, j] + 1, faces_uvs[i, j] + 1) for j in range(P)
                ]
            else:
                face = ["%d" % (faces[i, j] + 1) for j in range(P)]

            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)

            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)

######################################################
######################################################

app = Flask(__name__)

m_oath = MOUNT_PATH + "model/" + "model.pt"
if not os.path.isfile( m_oath ):
    import gdown
    url = 'https://drive.google.com/uc?id=1GPVPPqR_h2ZPOAsbttK93OxTU8DOyC_X'
    gdown.download(url, m_oath, quiet=False)

model = torch.jit.load(m_oath)
model.eval()

device = torch.device('cpu')
faces = torch.load(MOUNT_PATH + "data/" + 'faces.pt', map_location=device)
verts_uvs = torch.load(MOUNT_PATH + "data/" + 'uvs.pt', map_location=device)
faces_uvs_idx = torch.load(MOUNT_PATH + "data/" + 'faces_uvs_idx.pt', map_location=device)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'webp'])

print("Model Loaded!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/warmup', methods=['GET'])
def warmup():
    return "warming up!"

@app.route('/', methods=['POST'])
def predict():
    
    if request.method == 'POST':
    
        if 'file' not in request.files:
            return make_response(jsonify(error='No file!'), 404)

        file = request.files['file']

        if file.filename == '':
            return make_response(jsonify(error='No selected file!'), 404)

        if not allowed_file(file.filename):
            return make_response(jsonify(error='Invalid file type!'), 404)

        img = prepare_image( file )

        vertices, texture, _ = model( img.unsqueeze(0) ) 

        dirpath = tempfile.mkdtemp(dir='/tmp')

        @after_this_request
        def remove_file(response):
            shutil.rmtree(dirpath)
            return response

        f_name = str(uuid.uuid4())

        
        save_obj(
            f= os.path.join(dirpath, f_name + '.obj'),
            verts=vertices.squeeze(0).detach().cpu(),
            faces=faces.detach().cpu(),
            verts_uvs=verts_uvs.squeeze(0),
            faces_uvs=faces_uvs_idx,
            texture_map=texture.squeeze(0).permute(1, 2, 0),
            )

        
        zipfolder = zipfile.ZipFile(os.path.join(dirpath, f_name + '.zip'),'w', compression = zipfile.ZIP_STORED) # Compression type 

        # zip all the files which are inside in the folder
        for root,dirs, files in os.walk( dirpath ):
            for file in files:
                if ".zip" not in file:
                    zipfolder.write( os.path.join(dirpath, file), file)
        zipfolder.close()
        
        return send_file(os.path.join(dirpath, f_name + '.zip'),
            mimetype = 'application/zip',
            attachment_filename= f_name + '.zip',
            as_attachment = True)

@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)