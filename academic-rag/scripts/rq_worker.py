"""Run the RQ worker for PDF indexing jobs."""

import sys
from pathlib import Path

from loguru import logger
from rq import SimpleWorker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jobs.indexing import QUEUE_NAME, get_redis_connection, init_worker_pipeline


def main() -> None:
    connection = get_redis_connection("config.yaml")
    # 进程启动时加载一次模型，后续所有任务复用
    init_worker_pipeline("config.yaml")
    worker = SimpleWorker([QUEUE_NAME], connection=connection)
    logger.info("Starting RQ worker for queue '{}'", QUEUE_NAME)
    worker.work()


if __name__ == "__main__":
    main()
