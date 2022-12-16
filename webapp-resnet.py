"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import torch
from flask import Flask, render_template, request, redirect
from time import sleep
from torchvision import transforms

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_preprocessed = preprocess(img)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
        results = model(batch_img_tensor)
        _, index = torch.max(results, 1)
        percentage = torch.nn.functional.softmax(results, dim=1)[0] * 100
        result_text = labels[index[0]] + " " + str("{:.2f}".format(percentage[index[0]].item())) + "%"
        w, h = img.size
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('static/OpenSans-Regular.ttf', 32)
        draw.text((w/2, h/2), result_text,(252, 10, 10),font=font,anchor="mm",align='center')
        img.save('static/predict/out/image0.jpg')
        return redirect('static/predict/out/image0.jpg')

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number") 
    args = parser.parse_args()

    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    labels = ["hotdog", "not_hotdog"]
    # change your own model path below
    model = torch.load("models/ResNet152_model.pth", map_location=torch.device('cpu'))
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
