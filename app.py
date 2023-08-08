from fastapi import FastAPI
from fastapi import UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from stock_load_model import test_csv


app = FastAPI()
templates = Jinja2Templates(directory=".") # Change this path accordingly

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(file: UploadFile = File(...)):    

    file_bytes = file.file.read()

    try:
        contents = file_bytes
        with open("uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    output = test_csv("uploaded_" + file.filename)
    # TODO: Fix the label part
    return {"label": int(output[0][0] >= 0.8)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=8000, reload=True, access_log=False)
