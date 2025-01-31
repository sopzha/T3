# T3: Tuple Time Tree - Accurate and Fast Performance Prediction

This repository contains the code to reproduce all results of our paper about T3.

## Installation With Docker

Download the image from Docker Hub:
First clone this repository and move into the cloned directory.
```bash
sudo docker pull tupletimetree/t3
sudo docker run -v $(pwd):/app -it tupletimetree/t3
```

Or build the image yourself:
First clone this repository and move into the cloned directory.
```bash
sudo docker build -t t3 .
sudo docker run -v $(pwd):/app -it t3
```

## Native Installation

General Requirements:

```
sudo apt install lz4 python3 python3-venv python3-pip
```

Install python requirements:
First clone this repository and move into the cloned directory.

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

On MacOS you might need to

```bash
brew install libomp
```

Reproduce all figures of the paper:

```bash
. venv/bin/activate
python main.py
```

## Additional Benchmarks

The master script reproduces most results by default. However, some parts of the project are not portable. Most notably our database system only works on x86_64 Linux:

- **Join Order Microbenchmark and model latency**: Compiling our C++ benchmark file is only tested on x86_64 Linux.
  You can run this benchmark by adding the flag `-c`
- **Join Order Microbenchmark Query Testing**: Benchmarking the generated queries with different join orderings requires to run the database system. This only works on x86_64 Linux.
  You can run this benchmark by adding the flag `-j`
  This will download (about 6 GB) and generate (about 300 GB) the csv data and load it into the database (about 500 GB).
  Total required storage is about (800 GB).
- **Reproducing the full database benchmarks**: Creating the full dataset of benchmarked queries requires to run the database system. This only works on x86_64 Linux.
  You can run this benchmark by adding the flag `-b`
  This will download (about 6 GB) and generate (about 300 GB) the csv data and load it into the database (about 500 GB).
  Total required storage is about (800 GB).
  Benchmarks will take a while (about 8 hours on a 16 core machine)

The best way to run these additional benchmarks is to use the provided Dockerfile. To reproduce all results run:

```bash
sudo docker run -v $(pwd):/app -it tupletimetree/t3 -c -j
```

To re-create all results from scratch run:

```bash
sudo docker run -v $(pwd):/app -it tupletimetree/t3 -c -j -b
```

## Individual Figures
Each figure script has its own main function. These have to be run from the root of this directory. For example

```bash
. venv/bin/activate
python src/figures/latency_accuracy.py
```

