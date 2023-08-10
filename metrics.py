from prometheus_client import start_http_server, Counter, Gauge, Histogram
from prometheus_client import multiprocess
from prometheus_client import CollectorRegistry
import os
import psutil
import time
import random



def get_registry():
    if not  "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/"

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry




def start_metrics(port):
    start_http_server(port, registry=get_registry())




class Metrics:

    GET_counter = Counter(
        'GET_counter',
        'Counting get requests',
        registry=get_registry()
    )

    POST_counter = Counter(
        'POST_counter',
        'Counting post requests',
        registry=get_registry()
    )

    positive_pred_counter = Counter(
        'positive_pred_counter',
        'number of models positive prediction',
        registry=get_registry()
    )

    negative_pred_counter = Counter(
        'negative_pred_counter',
        'number of models negative prediction',
        registry=get_registry()
    )


    h = Histogram('request_latency_seconds',
               'latency of user request in seconds',
                buckets=[0.0001, 0.002, 0.03, 0.4, 1],
                registry=get_registry()
    )   
