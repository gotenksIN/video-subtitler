"""Process lock behavior for work-directory ownership."""

import os

import pytest

import gemini_subs


def test_second_owner_is_blocked_with_the_owner_pid(tmp_path):
    first = gemini_subs.acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=f"PID {os.getpid()}"):
            gemini_subs.acquire_lock(tmp_path)
    finally:
        gemini_subs.release_lock(first)


def test_lock_file_records_the_owner_pid(tmp_path):
    lock = gemini_subs.acquire_lock(tmp_path)
    try:
        assert (tmp_path / gemini_subs.LOCK_NAME).read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        gemini_subs.release_lock(lock)


def test_release_allows_a_new_owner(tmp_path):
    first = gemini_subs.acquire_lock(tmp_path)
    gemini_subs.release_lock(first)

    second = gemini_subs.acquire_lock(tmp_path)
    gemini_subs.release_lock(second)

    assert (tmp_path / gemini_subs.LOCK_NAME).exists()


def test_stale_pid_text_does_not_block_a_new_owner(tmp_path):
    (tmp_path / gemini_subs.LOCK_NAME).write_text("999999", encoding="utf-8")

    lock = gemini_subs.acquire_lock(tmp_path)
    try:
        assert (tmp_path / gemini_subs.LOCK_NAME).read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        gemini_subs.release_lock(lock)
