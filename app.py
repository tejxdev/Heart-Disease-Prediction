from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = model.predict(final)
    output = "Yes" if prediction[0] == 1 else "No"
    return render_template("result.html", prediction_text=f"Heart Disease: {output}")

if __name__ == "__main__":
    app.run(debug=True)
