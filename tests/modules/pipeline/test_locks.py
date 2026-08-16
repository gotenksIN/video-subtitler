"""Process lock behavior for work-directory ownership."""

import os

import pytest

from modules import pipeline


def test_second_owner_is_blocked_with_the_owner_pid(tmp_path):
    first = pipeline.acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=f"PID {os.getpid()}"):
            pipeline.acquire_lock(tmp_path)
    finally:
        pipeline.release_lock(first)


def test_lock_file_records_the_owner_pid(tmp_path):
    lock = pipeline.acquire_lock(tmp_path)
    try:
        assert (tmp_path / pipeline.LOCK_NAME).read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        pipeline.release_lock(lock)


def test_release_allows_a_new_owner(tmp_path):
    first = pipeline.acquire_lock(tmp_path)
    pipeline.release_lock(first)

    second = pipeline.acquire_lock(tmp_path)
    pipeline.release_lock(second)

    assert (tmp_path / pipeline.LOCK_NAME).exists()


def test_stale_pid_text_does_not_block_a_new_owner(tmp_path):
    (tmp_path / pipeline.LOCK_NAME).write_text("999999", encoding="utf-8")

    lock = pipeline.acquire_lock(tmp_path)
    try:
        assert (tmp_path / pipeline.LOCK_NAME).read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        pipeline.release_lock(lock)
