from flask import Flask, render_template, request
import os
import torch
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# YOLOv5 Model Load
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='best.pt'
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    results = model(input_path)

    results.render()

    output = results.ims[0]

    output_path = os.path.join(STATIC_FOLDER, "result.jpg")

    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    return render_template("index.html", image="result.jpg")


if __name__ == "__main__":
    app.run(debug=True)