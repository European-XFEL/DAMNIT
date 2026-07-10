import asyncio
import json
import logging
import signal
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from kafka.consumer import KafkaConsumer

from ..definitions import FILE_SUBMIT_TOPIC, update_brokers
from .service import notify_ready

log = logging.getLogger(__name__)

# We don't handle schema migrations for this. If changes are needed, stop the
# feeder service, let the queue drain, stop the main service, delete the DB
# and let it be
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS combine_tasks(
    id INTEGER PRIMARY KEY,
    lock_key INTEGER,
    assigned_to_worker INTEGER,
    failures INTEGER,
    task_info
)
"""

def feeder(db_path="combiner.sqlite"):
    """Fetch tasks from Kafka, feed them into the queue in SQLite"""
    stop = False

    def stop_on_signal(signum, frame):
        log.info("Stopping feeder on signal %d", signum)
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)

    db = sqlite3.connect(db_path, autocommit=True)
    db.executescript(DB_SCHEMA)

    cons = KafkaConsumer(
        FILE_SUBMIT_TOPIC,
        bootstrap_servers=update_brokers(),
        group_id='xfel-da-damnit-combiner',
        consumer_timeout_ms=600_000,
    )
    notify_ready()
    while not stop:
        for (tp, records) in cons.poll(timeout_ms=2000, update_offsets=False).items():
            insert_rows = []
            for record in records:
                d = json.loads(record.value.decode())
                d['submitted_at'] = record.timestamp / 1000
                dst_file = f"{d['damnit_dir']}/p{d['proposal']}_r{d['run']}.h5"
                insert_rows.append((dst_file, d))
            db.executemany(
                "INSERT INTO combine_tasks VALUES (NULL, ?, NULL, 0, ?)",
                insert_rows
            )
            cons.seek(tp, records[-1].offset + 1)


class CombinerManager:
    def __init__(self, n_workers=16, db_path="combiner.sqlite"):
        self.n_workers = n_workers
        self.db = sqlite3.connect(db_path, autocommit=True)
        self.db.executescript(DB_SCHEMA)
        # Only 1
        self.db.execute("""
            UPDATE combine_tasks SET assigned_to_worker=NULL
            WHERE assigned_to_worker IS NOT NULL
        """)
        self.socket_dir = TemporaryDirectory(prefix="damnit-combiner-")
        self.socket_path = Path(self.socket_dir.name) / "socket"
        self.workers = {}
        self.stopping = False

    def close(self):
        self.db.close()
        self.socket_dir.cleanup()

    @classmethod
    async def amain(cls, n_workers=16, db_path="combiner.sqlite"):
        self = cls(n_workers, db_path)

        try:
            server = await asyncio.start_unix_server(
                self.connection, self.socket_path
            )
            async with asyncio.TaskGroup() as worker_task_grp:
                await self.start_workers(worker_task_grp)
                notify_ready()
                async with server:
                    await server.serve_forever()
                    self.stopping = True
                    self.stop_workers()

        finally:
            self.close()

    async def start_workers(self, taskgrp: asyncio.TaskGroup):
        for worker_id in range(self.n_workers):
            process = await self._start_worker(worker_id)
            taskgrp.create_task(self.watch_worker(worker_id, process))

    async def _start_worker(self, worker_id: int):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "damnit.backend.combine",
        )
        self.workers[worker_id] = process
        log.debug("Started worker %d with PID %d", worker_id, process.pid)
        return

    async def watch_worker(self, worker_id, process):
        while True:
            pid = process.pid
            exitcode = await process.wait()
            if self.stopping:
                return

            log.info("Worker %d (PID %d) exited with status %d. Restarting.",
                     worker_id, pid, exitcode)
            self.worker_failed(worker_id)
            process = await self._start_worker(worker_id)

    def stop_workers(self):
        self.stopping = True
        for process in self.workers.values():
            process.terminate()  # SIGTERM

    async def connection(
            self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        b = await reader.readline()
        req = json.loads(b.decode())
        msg_type = req['msg_type']
        if msg_type == "next_task":
            rep = self.assign_next_task(req)
        elif msg_type ==  'task_complete':
            rep = self.task_complete(req)
        else:
            log.warning("Unexpected message type %r", msg_type)
            writer.close()
            return

        writer.write(json.dumps(rep, indent=None).encode())
        writer.write(b"\n")
        writer.close()
        await writer.wait_closed()

    def assign_next_task(self, req: dict):
        worker_id = req['worker_id']
        self.db.execute("BEGIN IMMEDIATE")
        try:
            task_id, task_info = self.db.execute("""
                SELECT id, task_info FROM combine_tasks
                WHERE assigned_to_worker IS NULL AND failures < 3 AND lock_key NOT IN (
                    SELECT lock_key FROM combine_tasks WHERE assigned_to_worker IS NOT NULL
                ) ORDER BY id LIMIT 1
            """)
            self.db.execute("UPDATE combine_tasks SET assigned_to_worker=? WHERE id=?",
                            (worker_id, task_id))
        except:
            self.db.execute("ROLLBACk")
            raise
        else:
            self.db.execute("COMMIT")

        return {"task_id": task_id, "task_info": json.loads(task_info)}

    def task_complete(self, req):
        task_id = req['task_id']
        self.db.execute("DELETE FROM combine_tasks WHERE id=?", (task_id))
        return {}

    def task_error(self, req):
        self.worker_failed(req['worker_id'])
        return {}

    def worker_failed(self, worker_id):
        r = self.db.execute("""
            UPDATE combine_tasks SET assigned_to_worker=NULL, failures=failures+1
            WHERE worker_id=? RETURNING task_id, failures
        """, (worker_id,)).fetchone()
        if r is not None:
            task_id, failures = r
            log.warning("Worker has failed %d times while executing task ID %s",
                        failures, task_id)


def main():
    asyncio.run(CombinerManager.amain())
