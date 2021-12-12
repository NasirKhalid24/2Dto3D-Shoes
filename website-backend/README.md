## Inference Backend - 2D to 3D Sneakers Project
## Nasir Khalid - 40192389

The following repo has app.py which contains all the code needed

```
pip install -r requirements.txt
python app.py
```

Running app.py will automatically download the pytorch model and its pretrained weights - it will start the flask server with the model on the cpu

You can manually download a .pt file containing the model and pretrained weights from here then load it in your own pytorch script

https://drive.google.com/uc?id=1GPVPPqR_h2ZPOAsbttK93OxTU8DOyC_X

Once the server starts you can then send an image to root (/) route through a POST request with the image as part of formdata 'file' (sample curl command below)

```console
curl -i -X POST \
   -H "Content-Type:multipart/form-data" \
   -F "file=@\"./YOUR_IMAGE_FILE_PATH_HERE\";type=application/YOUR_IMAGE_EXTENSION;filename=\"YOUR_IMAGE_FILE_PATH_HERE\"" \
 'YOUR_FLASK_SERVER_ROUTE_HERE'
```