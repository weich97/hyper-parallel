"""Launch this node's distributed ranks without importing Torch in the parent."""

import os
import signal
import subprocess
import sys
import time


def main():
    """Run this worker group and preserve the first failed exit."""
    port, size, *worker_args = sys.argv[1:]
    local_size = int(size)
    node_count = int(os.environ.get("NNODES", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    master_address = os.environ["MASTER_ADDR"] if node_count > 1 else "127.0.0.1"
    if not 0 <= node_rank < node_count or local_size <= 0:
        raise ValueError("Invalid distributed node rank or local world size")
    if worker_args[0] == "--module":
        worker_args[0] = "-m"
    workers = {}
    try:
        for local_rank in range(local_size):
            rank = node_rank * local_size + local_rank
            env = {
                **os.environ,
                "MASTER_ADDR": master_address,
                "MASTER_PORT": port,
                "WORLD_SIZE": str(node_count * local_size),
                "LOCAL_WORLD_SIZE": size,
                "RANK": str(rank),
                "LOCAL_RANK": str(local_rank),
                "GROUP_RANK": str(node_rank),
            }
            env.pop("TORCHELASTIC_USE_AGENT_STORE", None)
            workers[rank] = subprocess.Popen([sys.executable, "-u", *worker_args], env=env)
        while workers:
            for rank, process in list(workers.items()):
                status = process.poll()
                if status is None:
                    continue
                workers.pop(rank)
                print(f"WORKER_EXIT rank={rank} status={status}", flush=True)
                if status != 0:
                    return status if status > 0 else 128 - status
            if workers:
                time.sleep(0.2)
        return 0
    finally:
        for process in workers.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5
        for process in workers.values():
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def on_term(signum, _frame):
    """Trigger cooperative child cleanup on Slurm termination."""
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_term)
    raise SystemExit(main())

