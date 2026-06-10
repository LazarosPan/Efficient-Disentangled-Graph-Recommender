Easy Parallelization (https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)

Optuna supports multiple ways to run parallel optimization.

    Multi-thread optimization:

            You can run multiple trials in parallel within a single process using the n_jobs parameter in optimize().

    Multi-process optimization:

            You can run multiple processes sharing the same storage backend, such as RDB or a file.

    Multi-node optimization:

            You can run the same optimization study on multiple machines.

            If you need to perform optimization across thousands of processing nodes, you can use GrpcStorageProxy to run distributed optimization on multiple machines.

The following diagram shows which strategy is suitable for which use case.
digraph storage_selector { rankdir=LR; node [shape=box]; { rank=same; multithread; single_node; many_nodes; grpc_storage; } multithread [label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD>Multi-thread or Multi-process?</TD></TR> </TABLE> >]; single_node [label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD>Single node/<BR/>Multi-node?</TD></TR> </TABLE> >]; many_nodes [label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD>Do you need<BR/>a very large number of nodes?</TD></TR> </TABLE> >]; multithread_storages [ shape=box, style=rounded, href="#multi-thread-optimization", label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD><U>InMemoryStorage</U></TD></TR> <TR><TD><U>JournalStorage</U></TD></TR> </TABLE> > ]; singlenode_storages [ shape=box, style=rounded, href="#multi-process-optimization", label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD><U>JournalStorage</U></TD></TR> <TR><TD><U>RDBStorage</U></TD></TR> </TABLE> > ] rdb_storage [ shape=box, style=rounded, href="#multi-node-optimization", label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD><U>RDBStorage</U></TD></TR> </TABLE> > ] grpc_storage [ shape=box, style=rounded, href="#grpc-storage-proxy", label=< <TABLE BORDER="0" CELLBORDER="0" CELLALIGN="LEFT"> <TR><TD><U>GrpcStorageProxy</U></TD></TR> </TABLE> > ] multithread -> multithread_storages [label="Multi-thread"]; multithread -> single_node [label="Multi-process"]; single_node -> singlenode_storages [label="Single node"]; single_node -> many_nodes [label="Multi-node"]; many_nodes -> rdb_storage [label="No"]; many_nodes -> grpc_storage [label="Yes"]; }
Multi-thread Optimization

Note

Recommended backends:

        InMemoryStorage

        JournalStorage

        RDBStorage

You can run multiple trials in parallel just by setting the n_jobs parameter in optimize().

Multi-thread optimization has traditionally been inefficient in Python due to the Global Interpreter Lock (GIL). However, starting from Python 3.14 (pending official release), the GIL is expected to be removed. This change will make multi-threading a good option, especially for parallel optimization.

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial
import threading


def objective(trial: Trial):
    print(f"Running trial {trial.number=} in {threading.current_thread().name}")
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study(
    storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
)
study.optimize(objective, n_trials=20, n_jobs=4)

Multi-process Optimization with JournalStorage

Note

Recommended backends:

        JournalStorage

        RDBStorage

You can run multiple processes for optimization by using shared storage. Since InMemoryStorage is not designed to be shared across processes, it cannot be used for multi-process optimization.

The following example shows how to use JournalStorage for multi-process optimization with multiprocessing module.

import optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os


def objective(trial):
    print(f"Running trial {trial.number=} in process {os.getpid()}")
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def run_optimization(_):
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
        load_if_exists=True, # Useful for multi-process or multi-node optimization.
    )
    study.optimize(objective, n_trials=3)

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(run_optimization, range(12))


Multi-node Optimization with RDBStorage

Since JournalFileBackend uses file locks on the local filesystem, it operates safely for multiple processes on the same host. However, if accessed simultaneously from multiple machines via NFS (or similar), the file locks may not work correctly, which could lead to race conditions. it is likely to cause race conditions when accessed by multiple machines.

Therefore, for multi-node optimization, it is recommended to use RDBStorage. You can use MySQL, PostgreSQL, or other RDB backends.

For example, when using MySQL, you need to set up a MySQL server and create a database for Optuna.

$ mysql -u username -e "CREATE DATABASE IF NOT EXISTS example"

Then, you can use this MySQL database as a storage backend by setting the MySQL URL as the value of the storage parameter in create_study().

import optuna


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="distributed_test",
        storage="mysql://username:password@127.0.0.1:3306/example",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

Multi-node Optimization with GrpcStorageProxy

However, if you are running thousands of process nodes, an RDB server may not be able to handle the load. In that case, you can use GrpcStorageProxy to distribute the server load.

GrpcStorageProxy is a proxy storage layer that internally uses RDBStorage as its backend. It can efficiently handle high-throughput concurrent requests from multiple machines.

The following example shows how to use GrpcStorageProxy. Since GrpcStorageProxy is a proxy storage, you need to run a gRPC server with RDBStorage backend first.

from optuna.storages import run_grpc_proxy_server
from optuna.storages import get_storage

storage = get_storage("mysql+pymysql://username:password@127.0.0.1:3306/example")
run_grpc_proxy_server(storage, host="localhost", port=13000)

Then, on each machine, you can run the following code to connect to the gRPC proxy storage.

import optuna

from optuna.storages import GrpcStorageProxy


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    storage = GrpcStorageProxy(host="localhost", port=13000)
    study = optuna.create_study(
        study_name="grpc_proxy_multinode",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)
