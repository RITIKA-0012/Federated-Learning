from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "client": "client_2",
        "status": "online"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Temporary test prediction.
    # Later we will replace this with your XGBoost model.
    probability = 0.82

    return jsonify({
        "client": "client_2",
        "probability": probability,
        "status": "success"
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8102,
        debug=False
    )