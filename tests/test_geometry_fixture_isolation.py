"""A process-scoped geometry guard must not contaminate other pytest cases."""
import socket
import tempfile
from pathlib import Path

import pytest


def test_source_race_fixture_restores_network_entrypoints_for_following_workers(tmp_path):
    from test_native_point_geometry import test_source_import_guard_precedes_execution_and_layer_creation
    connect, create = socket.socket.connect, socket.create_connection
    with pytest.MonkeyPatch.context() as scoped:
        test_source_import_guard_precedes_execution_and_layer_creation(tmp_path, scoped)
    assert socket.socket.connect is connect
    assert socket.create_connection is create
    # Exercise the actual connection primitive used by forkserver workers.
    # AF_UNIX paths have a small OS limit; hosted/server scratch roots are long.
    with tempfile.TemporaryDirectory(prefix='eg-sock-') as short:
        address = str(Path(short)/'worker.sock')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(address); server.listen(1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(address)
                accepted, _ = server.accept()
                with accepted:
                    client.sendall(b'live')
                    assert accepted.recv(4) == b'live'
