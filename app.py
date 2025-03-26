from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static'

if not os.path.exists('uploads'):
    os.makedirs('uploads')

if not os.path.exists('static'):
    os.makedirs('static')

def enhance_image(image_path, brightness=50, contrast=30, denoise=False, edges=False, grayscale=False):
    img = cv2.imread(image_path)

    # Adjust brightness and contrast
    img = cv2.convertScaleAbs(img, alpha=1 + contrast / 100, beta=brightness)

    # Convert to grayscale if selected
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply noise reduction if selected
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Apply edge detection if selected
    if edges:
        img = cv2.Canny(img, 100, 200)

    # Save processed image
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], "enhanced.jpg")
    cv2.imwrite(processed_path, img)

    return "enhanced.jpg"

@app.route("/", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        brightness = int(request.form.get('brightness', 50))
        contrast = int(request.form.get('contrast', 30))
        denoise = 'denoise' in request.form
        edges = 'edges' in request.form
        grayscale = 'grayscale' in request.form

        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            processed_filename = enhance_image(img_path, brightness, contrast, denoise, edges, grayscale)
            return redirect(url_for("display_image", filename=processed_filename))

    return render_template("index.html")

@app.route("/display/<filename>")
def display_image(filename):
    return render_template("display.html", filename=filename)

@app.route("/download/<filename>")
def download_image(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
