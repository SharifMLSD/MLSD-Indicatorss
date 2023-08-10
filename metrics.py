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

    Request_counter = Counter(
        'Request_counter',
        'Counting get or post requests',
        ['method'],
        registry=get_registry()
    )

    pred_counter = Counter(
        'pred_counter',
        'number of models positive or negetive prediction',
        ['pred'],
        registry=get_registry()
    )

    h = Histogram('request_latency_seconds',
               'latency of user request in seconds',
                buckets=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                registry=get_registry()
    )   
