# MLSD-Indicatorss

In this repository, we implement a stock prediction app to suggest users buying or (selling/holding) their properties. The app is currently up at https://mlsd-indicatorss.darkube.app/.

## Using our model

You can deploy our model in your application. In order to do so, you will need to pass `modelV1.pth` and `stock_load_model.py`. You can also install requirements from `requierments.txt` file.
Using `test_csv(path)` from `stock_load_mode.py` you can read csv file from `path` and get model output. For more information on usage, see `app.py`.

