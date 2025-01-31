from time import sleep

from invoke import run


def start_webserver_new():
    run(f"setsid nohup sh -c './webserver benchmark_setup/db/all.db' > /dev/null 2>&1 & ", disown=True)
    sleep(2)


def kill_webserver_new():
    r1 = run(f"ps -ef | grep 'webserver benchmark_setup/db/all.db' | head -n 1 | awk '{{ print $2 }}'")
    pid = r1.stdout.strip()
    run(f"kill {pid}")
