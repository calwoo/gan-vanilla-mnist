"""
Script to download MNIST for personal use. Modified from the Stanford NLP course git. This is recreated
for my own personal learning.
"""

import os
import sys
import gzip
import argparse
import requests
import subprocess
from progress.bar import Bar

parser = argparse.ArgumentParser(description="downloader for MNIST.")
parser.add_argument("--dataset", type=str, choices=["mnist"],
    help="name of dataset to download")

def prepare_data_dir(path="./data"):
    # if there isn't a data folder, create one
    if not os.path.exists(path):
        os.mkdir(path)

def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, "mnist")
    if os.path.exists(data_dir):
        print("MNIST already found, aborting script.")
        return
    else:
        os.mkdir(data_dir)
    # now that we know we don't have it, download it from lecun's site
    url_base = "http://yann.lecun.com/exdb/mnist/"
    file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
    # downloading logic
    for filename in file_names:
        url = (url_base + filename).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, filename)
        print("Downloading ", filename)
        subprocess.call(["curl", url, "-o", out_path])
        print("Decompressing ", filename)
        subprocess.call(["gzip", "-d", out_path])

if __name__ == "__main__":
    args = parser.parse_args()
    prepare_data_dir()
    # download the data
    if "mnist" in args.dataset:
        download_mnist("./data")