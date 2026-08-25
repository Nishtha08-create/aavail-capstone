import time
from flask import Flask, request, jsonify
from model import predict
from logger import log_prediction

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    start_time = time.time()
    payload = request.get_json()
    
    if not payload or "input" not in payload:
        return jsonify({"error": "Invalid request. Must provide 'input' array."}), 400
    
    country = payload.get("country", "all")
    test_mode = payload.get("test_mode", False)
    input_data = payload["input"]
    
    if len(input_data) < 7:
        input_data = (input_data * 7)[:7]
    elif len(input_data) > 7:
        input_data = input_data[-7:]
        
    prediction = predict(input_data, test_mode=test_mode)
    runtime = time.time() - start_time
    
    log_prediction(input_data, prediction, runtime, country=country, test_mode=test_mode)
    
    return jsonify({
        "status": "success",
        "country": country,
        "prediction": round(float(prediction), 2),
        "runtime_seconds": round(runtime, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)