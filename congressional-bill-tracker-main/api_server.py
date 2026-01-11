import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import predict_cli


def _compute_overall(result):
    viability = result["viability"]["ensemble"]
    passage = None
    if result.get("passage") and result["passage"].get("ensemble") is not None:
        passage = result["passage"]["ensemble"]

    if result.get("has_become_law"):
        return 1.0, viability, passage
    if passage is None:
        return viability * 0.05, viability, passage
    return viability * passage, viability, passage


class PredictionHandler(BaseHTTPRequestHandler):
    server_version = "PredictCLI/1.0"

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            self._send_json(404, {"error": "Not found"})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        if self.path.rstrip("/") != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON payload"})
            return

        bill_number = str(payload.get("bill_number", "")).strip()
        bill_type = str(payload.get("bill_type", "")).strip().lower()
        congress_raw = payload.get("congress", 118)

        if not bill_number or not bill_type:
            self._send_json(
                400,
                {"error": "bill_number and bill_type are required"},
            )
            return

        try:
            congress = int(congress_raw)
        except (TypeError, ValueError):
            congress = 118

        try:
            result = predict_cli.predict_bill(bill_number, congress, bill_type)
            overall, viability, passage = _compute_overall(result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(
            200,
            {
                "overall": overall,
                "viability": viability,
                "passage": passage,
                "stage": result.get("stage"),
                "days_active": result.get("days_active"),
                "has_become_law": result.get("has_become_law"),
                "has_passed_house": result.get("has_passed_house"),
                "has_passed_senate": result.get("has_passed_senate"),
            },
        )


def main():
    parser = argparse.ArgumentParser(
        description="Serve pass predictions from predict_cli via HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), PredictionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
