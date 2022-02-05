from flask import Flask, jsonify, request
from  reader import prediction

app = Flask(__name__)
@app.route("/alphabet-pred", methods = ["POST"])

def predict_alphabet():
    image = request.files.get("V")
    pred = prediction(image)
    return jsonify({
        "Prediction of the alphabet shown" : pred
    }), 200

if __name__ == "__main__":
    app.run(debug = True)


