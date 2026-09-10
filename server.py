from typing import Dict, Any

from flask import Flask, request, jsonify

from scheduler import CloudScheduler
from aggregator import PredictionAggregator


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Replace this with your actual Cloudflare Quick Tunnel URL.

GATEWAY_URL = (
    "https://ent-pamela-nano-determine.trycloudflare.com"
)


MAX_CLIENTS = 5

RISK_THRESHOLD = 0.5


# ============================================================
# OBJECTS
# ============================================================

scheduler = CloudScheduler(
    GATEWAY_URL
)

aggregator = PredictionAggregator()


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def root():

    return jsonify({

        "service":
            "CloudScheduleAI",

        "status":
            "running",

        "service_type":
            "cloud_scheduler",

        "gateway":
            GATEWAY_URL,

        "max_clients":
            MAX_CLIENTS
    })


# ============================================================
# SERVER HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "service":
            "cloud_scheduler",

        "status":
            "online",

        "gateway":
            GATEWAY_URL
    })


# ============================================================
# CLIENT STATUS
# ============================================================

@app.route("/clients", methods=["GET"])
def clients_status():

    import asyncio

    result = asyncio.run(
        scheduler.schedule(
            max_clients=MAX_CLIENTS
        )
    )

    return jsonify(result)


# ============================================================
# MAIN PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    import asyncio

    patient_data = request.get_json(
        silent=True
    )

    if not patient_data:

        return jsonify({

            "status": "error",

            "message":
                "Patient data is required."

        }), 400

    print("\n")
    print("=" * 60)
    print("NEW HEART DISEASE PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # CHECK CLIENTS
    # --------------------------------------------------------

    scheduling_result = asyncio.run(
        scheduler.schedule(
            max_clients=MAX_CLIENTS
        )
    )

    print("\nSCHEDULER STATUS")

    print(
        "Gateway:",
        "ONLINE"
        if scheduling_result.get(
            "gateway_online"
        )
        else "OFFLINE"
    )

    available_clients = (
        scheduling_result.get(
            "available_clients",
            []
        )
    )

    selected_clients = (
        scheduling_result.get(
            "selected_clients",
            []
        )
    )

    print(
        "Available clients:",
        available_clients
    )

    print(
        "Selected clients:",
        selected_clients
    )

    # --------------------------------------------------------
    # NO CLIENTS
    # --------------------------------------------------------

    if not selected_clients:

        return jsonify({

            "status":
                "error",

            "message":
                "No clients are currently available.",

            "scheduler":
                scheduling_result

        }), 503

    # --------------------------------------------------------
    # STEP 2
    # REQUEST PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nRequesting predictions "
        "from selected clients..."
    )

    predictions = asyncio.run(

        scheduler.request_predictions(
            patient_data
        )
    )

    # --------------------------------------------------------
    # NO PREDICTIONS
    # --------------------------------------------------------

    if not predictions:

        return jsonify({

            "status":
                "error",

            "message":
                "No client returned a valid prediction.",

            "selected_clients":
                selected_clients

        }), 503

    # --------------------------------------------------------
    # STEP 3
    # PRINT CLIENT RESULTS
    # --------------------------------------------------------

    print("\nCLIENT PREDICTIONS")

    for prediction in predictions:

        print(

            f"{prediction['client']} "
            f"→ "
            f"{prediction['probability'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # STEP 4
    # ADD WEIGHTS
    # --------------------------------------------------------

    # For now every client has equal weight.
    #
    # Later you can give better-performing
    # models higher weights.

    for prediction in predictions:

        prediction["weight"] = 1.0

    # --------------------------------------------------------
    # STEP 5
    # AGGREGATION
    # --------------------------------------------------------

    final_probability = (

        aggregator.average(
            predictions
        )

    )

    # --------------------------------------------------------
    # STEP 6
    # CLASSIFICATION
    # --------------------------------------------------------

    risk = aggregator.classify(

        final_probability,

        RISK_THRESHOLD
    )

    # --------------------------------------------------------
    # STEP 7
    # SUMMARY
    # --------------------------------------------------------

    aggregation_summary = (

        aggregator.summary(

            predictions,

            final_probability

        )

    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Final Probability: "
        f"{final_probability * 100:.2f}%"
    )

    print(
        f"Risk: {risk}"
    )

    print(
        f"Participating Clients: "
        f"{len(predictions)}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return jsonify({

        "status":
            "success",

        "scheduler": {

            "gateway_online":
                scheduling_result.get(
                    "gateway_online"
                ),

            "total_clients":
                scheduling_result.get(
                    "total_clients"
                ),

            "available_clients":
                available_clients,

            "selected_clients":
                selected_clients,

            "successful_clients":
                [
                    p["client"]
                    for p in predictions
                ]
        },

        "client_predictions":
            predictions,

        "aggregation": {

            "method":
                "simple_average",

            "probability":
                round(
                    final_probability,
                    6
                ),

            "percentage":
                round(
                    final_probability * 100,
                    2
                ),

            "threshold":
                RISK_THRESHOLD,

            "risk":
                risk,

            "details":
                aggregation_summary
        }
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "CloudScheduleAI "
        "Cloud Scheduler"
    )
    print("=" * 60)

    print(
        "\nGateway:"
    )

    print(
        GATEWAY_URL
    )

    print(
        "\nServer:"
    )

    print(
        "http://127.0.0.1:8000"
    )

    print("=" * 60)

    app.run(

        host="127.0.0.1",

        port=8000,

        debug=False
    )