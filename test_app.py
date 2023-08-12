import requests
import time
from stock_load_model import test_model_ci

def test_get():
    get_url = "https://mlsd-indicatorss.darkube.app/"
    print("Testing GET method: ", get_url)
    response = requests.get(get_url)
    assert response.status_code == 200
    

def test_post():
    post_url = "https://mlsd-indicatorss.darkube.app/predict"
    print("Testing POST method: ", post_url)
    file_path = "vabemellat_30min_200.csv"
    with open(file_path , "rb") as file:
        files = {"file": file}
        response = requests.post(post_url, files=files)
    assert response.status_code == 200

def test_end_to_end_sample():
    post_url = "https://mlsd-indicatorss.darkube.app/predict"
    print("Testing E2E app: ", post_url)
    file_path = "vabemellat_30min_200.csv"
    with open(file_path , "rb") as file:
        files = {"file": file}
        response = requests.post(post_url, files=files)
    assert response.json()['label'] == 1


def test_model_accuracy():
    acc, precision = test_model_ci()
    print(acc, '=================')
    assert acc >= 0.5

if __name__=="__main__":
    test_get()
    test_post()
    test_end_to_end_sample()
    test_model_accuracy()
