import requests
import time

def test_latency():
    print("Testing ...")
    start_time = time.time()
    file_path = "vabemellat_30min_200.csv"
    with open(file_path , "rb") as file:
        files = {"file": file}
        response = requests.post("https://mlsd-indicatorss.darkube.app/predict", files=files)
    print("Response status: ", response.status_code, '=====================')
    assert response.status_code == 200
    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")

if __name__=="__main__":
    test_latency()
