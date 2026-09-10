from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ============================================================
# CLIENT CONFIGURATION
# ============================================================

CLIENTS = {
    "client_1": "http://127.0.0.1:8101",
    "client_2": "http://127.0.0.1:8102",
    "client_3": "http://127.0.0.1:8103",
    "client_4": "http://127.0.0.1:8104",
    "client_5": "http://127.0.0.1:8105",
}


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service": "CloudSheduleAI Client Gateway",
        "status": "online",
        "gateway_port": 8001,
        "clients": list(CLIENTS.keys())
    })


# ============================================================
# GATEWAY HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "service": "client_gateway",
        "status": "online",
        "clients": list(CLIENTS.keys())
    })


# ============================================================
# CHECK ALL CLIENTS
# ============================================================

@app.route("/clients", methods=["GET"])
def get_clients():

    result = {}

    for client_id, url in CLIENTS.items():

        try:

            response = requests.get(
                f"{url}/health",
                timeout=3
            )

            if response.status_code == 200:

                result[client_id] = {
                    "status": "online",
                    "url": url,
                    "response": response.json()
                }

            else:

                result[client_id] = {
                    "status": "offline",
                    "url": url,
                    "http_status": response.status_code
                }

        except requests.exceptions.RequestException as e:

            result[client_id] = {
                "status": "offline",
                "url": url,
                "error": str(e)
            }

    online_clients = [
        client_id
        for client_id, info in result.items()
        if info["status"] == "online"
    ]

    offline_clients = [
        client_id
        for client_id, info in result.items()
        if info["status"] == "offline"
    ]

    return jsonify({
        "total_clients": len(CLIENTS),
        "online_count": len(online_clients),
        "offline_count": len(offline_clients),
        "online_clients": online_clients,
        "offline_clients": offline_clients,
        "clients": result
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if data is None:

        return jsonify({
            "status": "error",
            "message": "Request body must contain valid JSON"
        }), 400

    results = {}

    for client_id, url in CLIENTS.items():

        try:

            response = requests.post(
                f"{url}/predict",
                json=data,
                timeout=10
            )

            if response.status_code == 200:

                results[client_id] = {
                    "status": "success",
                    "response": response.json()
                }

            else:

                results[client_id] = {
                    "status": "error",
                    "http_status": response.status_code,
                    "response": response.text
                }

        except requests.exceptions.Timeout:

            results[client_id] = {
                "status": "timeout",
                "message": "Client did not respond within 10 seconds"
            }

        except requests.exceptions.RequestException as e:

            results[client_id] = {
                "status": "offline",
                "error": str(e)
            }

    successful_clients = [
        client_id
        for client_id, info in results.items()
        if info["status"] == "success"
    ]

    failed_clients = [
        client_id
        for client_id, info in results.items()
        if info["status"] != "success"
    ]

    return jsonify({

        "status": "completed",

        "total_clients": len(CLIENTS),

        "successful_clients": successful_clients,

        "failed_clients": failed_clients,

        "results": results

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CloudSheduleAI Client Gateway")
    print("=" * 60)

    print("\nRegistered Clients:")

    for client_id, url in CLIENTS.items():
        print(f"  {client_id} -> {url}")

    print("\nGateway running on:")
    print("  http://127.0.0.1:8001")

    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=8001,
        debug=False
    )