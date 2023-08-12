from fastapi import FastAPI
from fastapi import UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os 
import threading
import psutil
from multiprocessing import Process


from stock_load_model import test_csv
from metrics import Metrics, start_metrics
import time





app = FastAPI(debug=True)
templates = Jinja2Templates(directory=".") # Change this path accordingly

@app.get("/", response_class=HTMLResponse)
def index(request: Request):

    Metrics.Request_counter.labels(method='GET').inc()
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(file: UploadFile = File(...)): 
    start = time.perf_counter()  

    file_bytes = file.file.read()
    
    try:
        contents = file_bytes
        with open("uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    

    Metrics.Request_counter.labels(method='POST').inc()
    
    output = test_csv("uploaded_" + file.filename)
    
    end = time.perf_counter() - start
    Metrics.h.observe(end)
    # TODO: Fix the label part

    label_pred = int(output[0][0] >= 0.8)
    if label_pred == 0:
        Metrics.pred_counter.labels(pred='NEGATIVE').inc()
    else:
        Metrics.pred_counter.labels(pred='POSITIVE').inc()

    return {"label": 'Buy!' if label_pred else "Don't buy!"}


if __name__ == "__main__":
    import uvicorn
    start_metrics(8080)

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, access_log=False, workers=1 )

    
