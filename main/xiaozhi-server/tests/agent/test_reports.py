"""待汇报箱：存取、过期、找设备空闲时念、设备连上后念。"""

import time
from types import SimpleNamespace

import pytest

from core.admin import connections
from core.agent import reports
from core.agent.grokbot_client import _sleep as real_sleep


@pytest.fixture(autouse=True)
def clean():
    reports.clear()
    yield
    reports.clear()


def test_add_take_expire_and_cap(monkeypatch):
    assert reports.take_text("dev") == "" and not reports.pending("dev")
    reports.add("dev", "bot-1", "孙悟空做完了：好。")
    reports.add("dev", "bot-2", "林黛玉做完了：嗯。")
    reports.add("dev", "bot-1", "   ")  # 空的不收
    assert reports.pending("dev") and reports.pending("dev", "bot-2") and not reports.pending("dev", "bot-9")
    assert reports.take_text("dev", "bot-1") == "孙悟空做完了：好。"
    assert reports.take_text("dev") == "林黛玉做完了：嗯。"
    assert not reports.pending("dev")
    # 过期
    reports.add("dev", "bot-1", "旧的。")
    real_time = time.time
    monkeypatch.setattr(reports.time, "time", lambda: real_time() + reports.EXPIRE_SECONDS + 1)
    assert reports.take_text("dev") == ""
    monkeypatch.undo()
    # 上限
    for i in range(reports.MAX_PER_DEVICE + 5):
        reports.add("dev", "bot-1", f"第 {i} 条。")
    assert len(reports.take("dev")) == reports.MAX_PER_DEVICE


class FakeConn:
    def __init__(self, loop=None, device="dev-1"):
        self.session_id, self.device_id = "s-1", device
        self.client_is_speaking, self.client_have_voice = False, False
        self.tts, self.loop = object(), loop
        self.client_abort, self.sentence_id = True, None
        self.logger = SimpleNamespace(bind=lambda **k: SimpleNamespace(info=lambda *a: None, warning=lambda *a: None))


@pytest.fixture
def spoken(monkeypatch):
    out = []
    import core.handle.intentHandler as intent_mod
    monkeypatch.setattr(intent_mod, "speak_txt", lambda conn, text: out.append(text))
    return out


async def test_deliver_waits_for_idle(spoken):
    conn = FakeConn()
    connections.register(conn)
    try:
        reports.add("dev-1", "bot-1", "孙悟空做完了：好。")
        conn.client_is_speaking = True

        async def free():
            await real_sleep(0.3)
            conn.client_is_speaking = False

        import asyncio
        task = asyncio.ensure_future(free())
        assert await reports.deliver_when_idle(conn, timeout=3) is True
        await task
        assert spoken == ["孙悟空做完了：好。"] and conn.client_abort is False and conn.sentence_id
        assert not reports.pending("dev-1")
        assert await reports.deliver_when_idle(conn, timeout=1) is False  # 没有了
    finally:
        connections.unregister(conn)


async def test_deliver_gives_up_when_offline_or_busy(spoken):
    conn = FakeConn()
    reports.add("dev-1", "bot-1", "孙悟空做完了：好。")
    assert await reports.deliver_when_idle(conn, timeout=1) is False  # 没登记在线
    assert reports.pending("dev-1") and spoken == []
    connections.register(conn)
    try:
        conn.client_is_speaking = True
        assert await reports.deliver_when_idle(conn, timeout=0.3) is False  # 一直在说话：留到下次
        assert reports.pending("dev-1") and spoken == []
    finally:
        connections.unregister(conn)


async def test_add_schedules_delivery_for_online_device(spoken):
    import asyncio
    conn = FakeConn(loop=asyncio.get_running_loop())
    connections.register(conn)
    try:
        reports.add("dev-1", "bot-1", "孙悟空做完了：好。")
        for _ in range(50):
            if spoken:
                break
            await real_sleep(0.02)
        assert spoken == ["孙悟空做完了：好。"]
    finally:
        connections.unregister(conn)


async def test_on_conn_open_waits_for_greeting(spoken, monkeypatch):
    import asyncio
    monkeypatch.setattr(reports, "AFTER_CONNECT_DELAY", 0.2)
    conn = FakeConn(loop=asyncio.get_running_loop())
    connections.register(conn)
    try:
        reports.add("dev-1", "bot-1", "孙悟空做完了：好。")  # 设备在说话时加的
        conn.client_is_speaking = True
        reports.on_conn_open(conn)
        await real_sleep(0.1)
        assert spoken == []
        conn.client_is_speaking = False
        for _ in range(50):
            if spoken:
                break
            await real_sleep(0.02)
        assert spoken == ["孙悟空做完了：好。"]
    finally:
        connections.unregister(conn)


async def test_speak_marks_device_speaking_so_second_report_waits(spoken):
    import asyncio
    conn = FakeConn(loop=asyncio.get_running_loop())
    connections.register(conn)
    try:
        reports.add("dev-1", "bot-1", "孙悟空做完了：第一。")
        for _ in range(50):
            if spoken:
                break
            await real_sleep(0.02)
        assert spoken == ["孙悟空做完了：第一。"] and conn.client_is_speaking is True  # 念的时候算在说话
        reports.add("dev-1", "bot-2", "林黛玉做完了：第二。")
        await real_sleep(0.2)
        assert spoken == ["孙悟空做完了：第一。"]  # 第一条还在念，第二条等着
        conn.client_is_speaking = False  # TTS 播完
        for _ in range(50):
            if len(spoken) == 2:
                break
            await real_sleep(0.02)
        assert spoken[-1] == "林黛玉做完了：第二。"
    finally:
        connections.unregister(conn)


async def test_quiet_device_waits_for_tap_or_wake(spoken):
    """睡眠时段：有待汇报也不主动念，只报条数；用户点了（deliver_now）照样念。"""
    import asyncio
    conn = FakeConn(loop=asyncio.get_running_loop())
    connections.register(conn)
    try:
        reports.set_quiet("dev-1", True)
        reports.add("dev-1", "bot-1", "孙悟空做完了：好。")
        reports.add("dev-1", "bot-2", "林黛玉做完了：嗯。")
        await real_sleep(0.1)
        assert spoken == [] and reports.count("dev-1") == 2
        assert reports.deliver_now("dev-1") is True
        for _ in range(50):
            if spoken:
                break
            await real_sleep(0.02)
        assert spoken == ["孙悟空做完了：好。林黛玉做完了：嗯。"] and reports.count("dev-1") == 0
        # 离开睡眠时段后照常主动念
        reports.set_quiet("dev-1", False)
        conn.client_is_speaking = False
        reports.add("dev-1", "bot-1", "又做完一件。")
        for _ in range(50):
            if len(spoken) == 2:
                break
            await real_sleep(0.02)
        assert spoken[-1] == "又做完一件。"
    finally:
        connections.unregister(conn)


def test_deliver_now_needs_online_device():
    reports.add("dev-9", "bot-1", "孙悟空做完了：好。")
    assert reports.deliver_now("dev-9") is False and reports.count("dev-9") == 1
