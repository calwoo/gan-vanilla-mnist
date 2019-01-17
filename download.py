"""
Script to download MNIST for personal use. Modified from the Stanford NLP course git. This is recreated
for my own personal learning.
"""

import os
import sys
import gzip
import argparse
import requests
from progress.bar import Bar

parser = argparse.ArgumentParser(description="downloader for MNIST.")
parser.add_argument("datasets", type=str, choices=["mnist"],
    help="name of dataset to download")

def prepare_data_dir(path="./data"):
    # if there isn't a data folder, create one
    if not os.path.exists(path):
        os.mkdir(path)

def download_mnist(dirpath):
    pass










if __name__ == "__main__":
    args = parser.parse_args()
    prepare_data_dir()
    # download the data
    if "mnist" in args.datasets:
        download_mnist("./data")