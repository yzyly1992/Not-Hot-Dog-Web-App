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
        results = model(img, size=224)
        if results.pandas().xyxy[0].empty:
            w, h = img.size
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('static/OpenSans-Regular.ttf', 32)
            draw.text((w/2, h/2),"Not Hot Dog",(252, 10, 10),font=font,anchor="mm",align='center')
            img.save('static/predict/out/image0.jpg')
        else:
            results.render()
            results.save(save_dir="static/predict/out", exist_ok='store_true')  
        return redirect('static/predict/out/image0.jpg')

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number") 
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
