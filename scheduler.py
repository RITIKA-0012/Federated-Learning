import asyncio
import time
from typing import Dict, List

import httpx


class CloudScheduler:

    def __init__(self, gateway_url: str):

        self.gateway_url = gateway_url.rstrip("/")

    # ============================================================
    # CHECK GATEWAY
    # ============================================================

    async def check_gateway(self) -> Dict:

        start_time = time.perf_counter()

        result = {
            "online": False,
            "latency_ms": None
        }

        try:

            async with httpx.AsyncClient() as http:

                response = await http.get(
                    f"{self.gateway_url}/health",
                    timeout=10.0
                )

            latency = (
                time.perf_counter() -
                start_time
            ) * 1000

            result["latency_ms"] = round(
                latency,
                2
            )

            if response.status_code == 200:

                result["online"] = True
                result["response"] = response.json()

            else:

                result["error"] = (
                    f"HTTP {response.status_code}"
                )

        except Exception as e:

            result["error"] = str(e)

        return result

    # ============================================================
    # GET CLIENT STATUS FROM GATEWAY
    # ============================================================

    async def get_client_status(self) -> Dict:

        try:

            async with httpx.AsyncClient() as http:

                response = await http.get(
                    f"{self.gateway_url}/clients",
                    timeout=15.0
                )

            if response.status_code != 200:

                return {
                    "status": "error",
                    "message": (
                        f"Gateway returned "
                        f"HTTP {response.status_code}"
                    )
                }

            return response.json()

        except Exception as e:

            return {
                "status": "error",
                "message": str(e)
            }

    # ============================================================
    # SCHEDULING
    # ============================================================

    async def schedule(
        self,
        max_clients: int = 5
    ) -> Dict:

        gateway = await self.check_gateway()

        if not gateway["online"]:

            return {
                "gateway_online": False,
                "gateway_latency_ms":
                    gateway.get("latency_ms"),

                "available_clients": [],
                "selected_clients": [],

                "total_clients": 5,
                "available_count": 0,
                "selected_count": 0,

                "error":
                    gateway.get(
                        "error",
                        "Gateway offline"
                    )
            }

        client_status = (
            await self.get_client_status()
        )

        if client_status.get("status") == "error":

            return {
                "gateway_online": True,

                "available_clients": [],
                "selected_clients": [],

                "total_clients": 5,
                "available_count": 0,
                "selected_count": 0,

                "error":
                    client_status.get(
                        "message"
                    )
            }

        online_clients = (
            client_status.get(
                "online_clients",
                []
            )
        )

        # Limit number of participating clients
        selected_clients = online_clients[
            :max_clients
        ]

        return {

            "gateway_online": True,

            "gateway_latency_ms":
                gateway.get("latency_ms"),

            "available_clients":
                online_clients,

            "selected_clients":
                selected_clients,

            "total_clients":
                client_status.get(
                    "total_clients",
                    5
                ),

            "available_count":
                len(online_clients),

            "selected_count":
                len(selected_clients)
        }

    # ============================================================
    # REQUEST PREDICTION THROUGH GATEWAY
    # ============================================================

    async def request_predictions(
        self,
        patient_data: Dict
    ) -> List[Dict]:

        start_time = time.perf_counter()

        try:

            async with httpx.AsyncClient() as http:

                response = await http.post(

                    f"{self.gateway_url}/predict",

                    json=patient_data,

                    timeout=60.0
                )

            total_latency = (
                time.perf_counter() -
                start_time
            ) * 1000

            if response.status_code != 200:

                print(
                    "Gateway prediction failed:",
                    response.status_code
                )

                return []

            data = response.json()

            gateway_results = data.get(
                "results",
                {}
            )

            predictions = []

            for client_id, result in (
                gateway_results.items()
            ):

                # Gateway returns:
                #
                # {
                #   "status": "success",
                #   "response": {
                #       "probability": 0.82
                #   }
                # }

                if result.get("status") != "success":
                    continue

                client_response = result.get(
                    "response",
                    {}
                )

                probability = client_response.get(
                    "probability"
                )

                if probability is None:
                    continue

                probability = float(
                    probability
                )

                probability = max(
                    0.0,
                    min(1.0, probability)
                )

                predictions.append({

                    "client": client_id,

                    "probability":
                        probability,

                    "status": "success",

                    "gateway_latency_ms":
                        round(
                            total_latency,
                            2
                        )
                })

            return predictions

        except Exception as e:

            print(
                "Prediction request error:",
                e
            )

            return []