import os
import subprocess
from pathlib import Path

import duckdb
import lz4.frame
import requests


def download_t3_file(filename: str):
    Path("downloaded_data").mkdir(parents=True, exist_ok=True)
    if not Path(f"downloaded_data/{filename}").exists():
        with requests.get(f"https://f003.backblazeb2.com/file/tuple-time-tree/{filename}", stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses
            with open(f"downloaded_data/{filename}", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


def gen_tpcds(sf: int, path: Path):
    Path("tpcds_db.db").unlink(missing_ok=True)
    con = duckdb.connect("tpcds_db.db")
    con.execute("INSTALL tpcds;")
    con.execute("LOAD tpcds;")
    con.execute(f"CALL dsdgen(sf = {sf});")
    tpcds_tables = [
        "call_center",
        "catalog_page",
        "catalog_sales",
        "catalog_returns",
        "customer",
        "customer_address",
        "customer_demographics",
        "date_dim",
        "household_demographics",
        "income_band",
        "inventory",
        "item",
        "promotion",
        "reason",
        "ship_mode",
        "store",
        "store_sales",
        "store_returns",
        "time_dim",
        "warehouse",
        "web_page",
        "web_sales",
        "web_returns",
        "web_site",
    ]
    dir = path / f"tpcds/sf{sf}"
    dir.mkdir(exist_ok=True, parents=True)
    for table in tpcds_tables:
        con.execute(f"COPY {table} TO '{dir}/{table}.dat' ( DELIMITER '|', HEADER FALSE);")


def gen_tpch(sf: int, path: Path):
    Path("tpch_db.db").unlink(missing_ok=True)
    con = duckdb.connect("tpch_db.db")
    con.execute("INSTALL tpch;")
    con.execute("LOAD tpch;")
    con.execute(f"CALL dbgen(sf = {sf});")
    tpch_tables = [
        "part",
        "region",
        "nation",
        "supplier",
        "partsupp",
        "customer",
        "orders",
        "lineitem",
    ]

    dir = path / f"tpch/sf{sf}"
    dir.mkdir(exist_ok=True, parents=True)
    for table in tpch_tables:
        con.execute(f"COPY {table} TO '{dir}/{table}.tbl' (DELIMITER '|', HEADER FALSE);")


def download_csvs():
    print("Downloading Datasets (6 GB)")
    download_t3_file("csvs.tar.lz4")

    print("Extracting Datasets (15 GB)")
    subprocess.run(
        ["tar", "--use-compress-program=lz4", "-xvf", "downloaded_data/csvs.tar.lz4", "-C", "benchmark_setup"],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
    )


def create_tpc_data():
    for sf in [1, 10, 100]:
        print(f"Generating TPC-H sf{sf} dataset ({sf} GB)")
        gen_tpch(sf, Path("benchmark_setup/csvs"))
        print(f"Generating TPC-DS sf{sf} dataset ({sf} GB)")
        gen_tpcds(sf, Path("benchmark_setup/csvs"))


def extract_sql():
    out_path = "benchmark_setup/sql"
    if not Path(out_path).exists():
        download_t3_file("sql.lz4")
        with open("downloaded_data/sql.lz4", "rb") as server:
            decompressed_data = lz4.frame.decompress(server.read())
        with open(out_path, "wb") as server:
            server.write(decompressed_data)
        subprocess.run(["chmod", "+x", out_path], cwd=os.getcwd())


def load_csvs_to_db():
    if not Path("benchmark_setup/csvs/accident").exists():
        print("Could not access CSV files")
        return
    print("Downloading SQL tool")
    download_t3_file("sql.lz4")
    extract_sql()
    print("Loading CSVs into db file (400 GB)")
    subprocess.run(["bash", "benchmark_setup/scripts/load_data.sh"], cwd=os.getcwd())
