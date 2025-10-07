from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# مدل اول: تشخیص نما
model_view = load_model("view_classifier(s2-model1).keras")
feature_extractor_view = Sequential(model_view.layers[:-1]) 

# مدل نهایی: تشخیص نوع تومور
final_model = load_model("final-model+pre-s2.keras")

# کلاس‌ها
tumor_classes = ["glioma", "meningioma", "notumor","pituitary"]
view_classes = ["axial", "coronal", "sagittal"]
IMAGE_SIZE = 128

def predict_tumor_and_view(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    view_pred = model_view.predict(img_array)
    view_idx = np.argmax(view_pred, axis=1)[0]
    view_conf = np.max(view_pred)
    view_label = view_classes[view_idx]

    view_feature = feature_extractor_view(img_array, training=False).numpy()

    tumor_pred = final_model.predict([img_array, view_feature])
    tumor_idx = np.argmax(tumor_pred, axis=1)[0]
    tumor_conf = np.max(tumor_pred)
    tumor_label = tumor_classes[tumor_idx]

    if tumor_label == "notumor":
        result = f"No Tumor, View: {view_label}"
    else:
        result = f"Tumor: {tumor_label}, View: {view_label}"

    return result, tumor_conf, view_conf

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    summary = {
        "total": 0,
        "no_tumor": 0,
        "tumors": {}
    }

    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, tumor_conf, view_conf = predict_tumor_and_view(filepath)
            results.append({
                'filename': file.filename,
                'result': result,
                'tumor_confidence': f"{tumor_conf * 100:.2f}%",
                'view_confidence': f"{view_conf * 100:.2f}%",
                'filepath': f'/uploads/{file.filename}'
            })

            summary["total"] += 1
            if "No Tumor" in result:
                summary["no_tumor"] += 1
            else:
                tumor_type = result.split(",")[0].split(": ")[1]
                summary["tumors"][tumor_type] = summary["tumors"].get(tumor_type, 0) + 1

    return render_template('index.html', results=results, summary=summary)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    from threading import Timer
    import webbrowser
    Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=True)
