"""[xiaozhi-hub] 回声识别过滤：回复播放期间和播完后几秒内，识别结果大部分是回复自己的字就丢掉。"""

import struct
from types import SimpleNamespace

from core.utils import reply_echo


def make_conn():
    return SimpleNamespace(sentence_id="s1", client_is_speaking=True)


def test_echo_of_reply_is_dropped_while_speaking():
    conn = make_conn()
    assert not reply_echo.during_reply(conn) and not reply_echo.is_echo(conn, "也是。")  # 还没有回复
    reply_echo.note_reply(conn, "我觉得也是，")
    reply_echo.note_reply(conn, "明天会下雨。")
    assert reply_echo.during_reply(conn)
    assert reply_echo.is_echo(conn, "也是。")
    assert reply_echo.is_echo(conn, "明天会下鱼")  # 认错一个字也算
    assert not reply_echo.is_echo(conn, "停一下，帮我查查股票")  # 用户自己的话
    assert not reply_echo.is_echo(conn, "。")


def test_window_after_reply_and_new_reply_resets(monkeypatch):
    conn = make_conn()
    now = [100.0]
    monkeypatch.setattr(reply_echo.time, "monotonic", lambda: now[0])
    reply_echo.note_reply(conn, "今天天气不错。")
    reply_echo.note_reply(conn, None)  # 这轮的音频发完了
    conn.client_is_speaking = False
    now[0] += reply_echo.WINDOW_AFTER - 0.5
    assert reply_echo.is_echo(conn, "天气不错")  # 设备还在播缓冲，回声还会来
    now[0] += 1
    assert not reply_echo.during_reply(conn) and not reply_echo.is_echo(conn, "天气不错")  # 过了窗口算用户说的
    # 新的一轮回复：上一轮的字不再算
    conn.sentence_id, conn.client_is_speaking = "s2", True
    reply_echo.note_reply(conn, "好的。")
    assert not reply_echo.is_echo(conn, "天气不错") and reply_echo.is_echo(conn, "好的")


def test_level_db():
    loud = struct.pack("<960h", *([16384] * 960))  # 半幅方波 ≈ -6 dBFS
    quiet = struct.pack("<960h", *([0] * 960))
    assert round(reply_echo.level_db([quiet, loud])) == -6
    assert reply_echo.level_db([]) == -120.0
