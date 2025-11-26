from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from predict import predict_image_safe_v2

app = FastAPI(title="Mushroom Classifier API")

CLASSES = [
    "contamination_bacterialblotch",
    "contamination_cobweb",
    "contamination_greenmold",
    "healthy_bag",
    "healthy_mushroom",
    "not_mushroom"
]

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        pred_class, maxp, m_dist, probs = predict_image_safe_v2(img_bgr)

        prob_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}

        return {
            "predicted_class": pred_class,
            "max_probability": float(maxp),
            "mahalanobis_distance": float(m_dist),
            "probabilities": prob_dict
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
