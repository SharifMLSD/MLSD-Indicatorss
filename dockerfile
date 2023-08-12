from python:3.8

RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

ENV DIR /opt/image-service

RUN mkdir $DIR

WORKDIR $DIR

COPY requirements.txt $DIR/

RUN pip install -r requirements.txt

COPY Makefile $DIR/

COPY app.py $DIR/
COPY metrics.py $DIR/
COPY stock_load_model.py $DIR/
COPY test_app.py $DIR/

COPY index.html $DIR/

COPY data/ $DIR/data

COPY vabemellat_30min_200.csv $DIR/

COPY modelV2.pth $DIR/

COPY modelV1.pth $DIR/

CMD python app.py


