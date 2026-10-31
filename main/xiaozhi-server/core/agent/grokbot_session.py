"""Grok Bot 会话：对着一个云端 Bot 的主对话发一句话，把它的回复一条条交给设备播报。

和 ClaudeSession 不同，这里没有本机进程：对话记录在 Bot 的云端电脑上，天然跨连接接续，
"会话 ID"就是 Bot 的 id。一轮的经过：

1. 先记下对话尾部已有的条目（之后只播新的）；
2. sendPrompt 发送，结果不明确时用 promptAcceptanceStatus 确认，不盲目重发；
3. 从事件流收：Bot 开始处理（agent-upserted）→ 我们的消息回显（transcript）→ 回复条目 → Bot 空闲；
   收到过回复、空闲 1.5 秒无新动静才算本轮结束。一条回复都没有时空闲不算数：Bot 想得久时会中途显示空闲，
   实测（2026-11-01）空闲 63 秒后才发出回复；这时一直等到回复，或空闲满 no_reply_timeout 转后台；
4. 兜底：发送后 10 秒没看到回显就让中枢重连并补拉尾部；等待期间每 12 秒轮询一次（listAgents 看状态 + 拉尾部）。
   推送和轮询按条目 id 去重，谁先到用谁。

用户拍板：多条回复每条到了就播；打断只停止播报，云端任务继续做，不找停止接口。
这一轮在设备这边结束时，留一个后台任务继续盯着这个 Bot，之后到的回复交给 on_report（见 core/agent/reports.py），
设备空闲时再念：被打断、等超时、设备断开、一直没回话时盯最多 2 小时；正常做完也再盯 10 分钟，接住它之后补发的
（先回一句"我去查一下"、过一会儿才发结果）。
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Dict, List, Optional

from core.agent.agent_session import Activity, AgentEvent, AgentSession, AgentSessionError, Done, Error, ErrorKind, TextDelta
from core.agent.grokbot_client import GrokBotError, GrokBotHub, Subscription

SENTENCE_END = set("。！？!?；;…~～.\n")
WAITING_NOTICE = "Grok Bot 在等你去 App 里处理。"
NO_REPLY_NOTICE = "Grok Bot 还没回话，回了我再告诉你。"

# 出错时播报给用户的话（代替 agent_session 里 Claude 的默认文案）
VOICE_MESSAGES = {
    ErrorKind.AUTH: "Grok Bot 还没登录，或者登录过期了，请到管理后台重新登录。",
    ErrorKind.SERVER: "Grok Bot 那边暂时连不上，稍后再试吧。",
    ErrorKind.TIMEOUT: "Grok Bot 这件事做得太久，我先不等了。它做完后我再告诉你。",
    ErrorKind.STALLED: "Grok Bot 一直没有回应，我先停下了。",
    ErrorKind.PROCESS_EXITED: "和 Grok Bot 的连接断开了，再说一次试试。",
    ErrorKind.UNKNOWN: "出了点小问题，再说一次试试。",
}


def voice_for(kind: ErrorKind) -> str:
    return VOICE_MESSAGES.get(kind) or VOICE_MESSAGES[ErrorKind.UNKNOWN]


@dataclass
class GrokBotSessionConfig:
    bot_id: str
    bot_name: str = ""
    voice_hint: str = ""  # 附在每句话后面的提示，让 Bot 口语化简短回答；留空不加
    echo_timeout: float = 10  # 发送后这么久没在事件流里看到回显，就重连并补拉
    poll_interval: float = 12  # 等回复期间轮询对话尾部的间隔
    done_delay: float = 1.5  # 收到过回复、Bot 空闲后再等这么久没新动静才算本轮结束
    no_reply_timeout: float = 90  # 消息送到了、一条回复都没有、Bot 连续空闲这么久：设备这边先结束，转后台盯着
    connect_wait: float = 3  # 发送前最多等事件流连上这么久
    watch_max: float = 2 * 3600  # 还没拿到回复就结束了（打断、超时、断开、没回话）：最多在后台盯多久
    linger_max: float = 600  # 正常做完后再盯多久，接住之后补发的消息；每补到一批再顺延这么久
    tail_limit: int = 20


def plain_text(text: str) -> str:
    """去掉 Markdown 记号（Bot 的回复是给屏幕看的），只留要念的文字。"""
    t = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    t = re.sub(r"`([^`\n]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__|~~)(.+?)\1", r"\2", t, flags=re.S)
    t = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", t)
    t = re.sub(r"^\s*\|?[\s:|-]+\|\s*$", "", t, flags=re.M)  # 表格分隔行
    t = t.replace("|", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    return t.strip()


def spoken_form(text: str) -> str:
    """一条回复整理成播报形式：去 Markdown，结尾补句号，TTS 才会马上念出来。"""
    t = plain_text(text)
    if t and t[-1] not in SENTENCE_END:
        t += "。"
    return t


def reply_text(entry: dict) -> Optional[str]:
    """Bot 给人看的文字：assistant 的 message，或 send-message 里的 text。其他条目（工具调用、通知）返回 None。"""
    kind = entry.get("kind")
    if kind == "message" and (entry.get("role") == "assistant" or entry.get("fromAgent")):
        return entry.get("content") or ""
    if kind == "send-message":
        m = entry.get("message") or {}
        if m.get("type") == "text":
            return m.get("content") or m.get("text") or ""
    return None


def agent_busy(agent: dict) -> bool:
    return bool(agent.get("isRunning") or agent.get("isRunningTurn") or agent.get("isComposingMessage"))


def agent_waiting(agent: dict) -> bool:
    return agent.get("awaitingUserResponse") not in (None, False, "", "null")


def is_streaming(entry: dict) -> bool:
    return bool(entry.get("isStreaming") or entry.get("streaming"))


class Tracker:
    """一轮里对 Bot 动静的判断：哪些条目要念、Bot 忙不忙、这轮算不算结束。一轮一个；被打断后交给后台任务接着用。"""

    def __init__(self, cfg: GrokBotSessionConfig, nonce: str, sent_text: str, seen: Dict[str, str], generation: int,
                 log: Callable[[str, str], None]):
        self.cfg = cfg
        self.nonce = nonce
        self.sent_text = sent_text
        self.seen = seen  # 条目 id -> 已播报的文字；发送前已有的条目也记在这里（值为空）
        self.raw: Dict[str, str] = {}  # 已播报条目的纯文字（没补句号），条目被更新、文字变长时只念多出来的部分
        self.generation = generation
        self.log = log
        self.t0 = time.monotonic()
        self.have_baseline = True  # 发送前读到了对话尾部；读不到时靠回显的 seq 分新旧
        self.echo_seq: Optional[int] = None  # 我们这句话在对话里的序号，比它小的都是旧的
        self.echoed = False
        self.busy = False
        self.was_busy = False
        self.idle_since: Optional[float] = None
        self.spoken = 0
        self.replied = False  # 这一段（本轮，或转后台后的每一批）收到过回复；没有回复时 Bot 空闲不算做完
        self.waiting = False
        self.recovered = False  # 已经因为没回显重连补拉过

    def _since(self) -> str:
        return f"+{time.monotonic() - self.t0:.1f}s"

    def entry(self, entry: dict) -> List[str]:
        """新条目：回显就标记，Bot 的文字就返回要念的话。"""
        entry_id = str(entry.get("id") or "")
        if not entry_id or is_streaming(entry):
            return []
        if entry.get("clientNonce") == self.nonce or (
                entry.get("kind") == "message" and entry.get("role") == "user" and entry.get("content") == self.sent_text):
            if not self.echoed:
                self.echoed = True
                if isinstance(entry.get("seq"), (int, float)):
                    self.echo_seq = int(entry["seq"])
                self.log("info", f"{self._since()} 消息已出现在对话里")
            self.seen.setdefault(entry_id, "")
            return []
        seq = entry.get("seq")
        if self.echo_seq is not None and isinstance(seq, (int, float)) and seq <= self.echo_seq:
            self.seen.setdefault(entry_id, "")  # 排在我们这句话前面的，都是旧的
            return []
        if not self.have_baseline and self.echo_seq is None:
            return []  # 没有基线又还没见到回显：分不清新旧，先不念（回显到了之后轮询会再送来）
        content = reply_text(entry)
        if entry_id in self.seen:
            # 已经念过的条目被更新：文字在原来基础上变长（流式追加）就只念多出来的部分，别的改动不管
            old = self.raw.get(entry_id)
            new = plain_text(content) if content is not None else ""
            if old is None or len(new) <= len(old) or not new.startswith(old):
                return []
            self.raw[entry_id] = new
            extra = spoken_form(new[len(old):].lstrip("。.！!？?；;，,、 \n"))  # 上一段已经补过句号
            if not extra:
                return []
            self.idle_since = None if self.busy else time.monotonic()
            self.replied = True
            self.log("info", f"{self._since()} 回复追加：{extra[:80]}")
            return [extra]
        if content is None:
            self.seen[entry_id] = ""
            return []
        spoken = spoken_form(content)
        self.seen[entry_id] = spoken
        if not spoken:
            return []
        self.raw[entry_id] = plain_text(content)
        self.spoken += 1
        self.replied = True
        self.idle_since = None if self.busy else time.monotonic()
        self.log("info", f"{self._since()} 回复：{spoken[:80]}")
        return [spoken]

    def agent(self, agent: dict) -> List[str]:
        """Bot 状态变化。第一次发现它在等用户时返回提示语。"""
        busy = agent_busy(agent)
        out: List[str] = []
        if busy and not self.was_busy:
            self.log("info", f"{self._since()} Bot 开始处理")
        elif busy != self.busy and self.was_busy:
            # 忙闲来回切换记下来：一条回复都没有时 Bot 中途变空闲，是"回复丢了"的根因，排查要看
            self.log("info", f"{self._since()} Bot {'又忙起来' if busy else '变空闲'}（已收 {self.spoken} 条回复）")
        if busy:
            self.was_busy = True
            self.idle_since = None
        elif self.busy or self.idle_since is None:
            self.idle_since = time.monotonic()
        self.busy = busy
        if agent_waiting(agent) and not self.waiting:
            self.waiting = True
            self.log("info", "Bot 在等用户处理（审批/提问），需要去 App 里看")
            out.append(WAITING_NOTICE)
        return out

    @property
    def idle_pending(self) -> bool:
        """收到过回复、Bot 已空闲：再等 done_delay 没新动静就算做完。"""
        return self.replied and self.idle_since is not None

    def finished(self, now: float) -> Optional[str]:
        """这轮结束了吗：done（Bot 做完）、waiting（在等用户去 App 处理）、None（还没）。
        一条回复都没有时 Bot 空闲不算做完：它想得久时会中途显示空闲，过一会儿才发回复。"""
        if self.idle_pending and now - self.idle_since >= self.cfg.done_delay:
            return "done"
        if self.waiting:
            return "waiting"
        return None

    def no_reply(self, now: float) -> bool:
        """消息送到了，一条回复都没有，Bot 连续空闲（没忙过就从发送算起）满 no_reply_timeout。"""
        if not self.echoed or self.replied or self.busy:
            return False
        since = self.idle_since if self.was_busy and self.idle_since is not None else self.t0
        return now - since >= self.cfg.no_reply_timeout

    def next_batch(self, replied: bool = False) -> None:
        """转后台盯着、或后台汇报完一批后：之后要等新的回复才算做完。replied 表示手上已有没念的回复。"""
        self.replied = replied


class GrokBotSession(AgentSession):
    def __init__(self, hub: GrokBotHub, config: GrokBotSessionConfig, logger=None,
                 on_report: Callable[[str], None] = None, device_key: str = ""):
        self.hub = hub
        self.config = config
        self.logger = logger
        self.on_report = on_report  # 被打断后 Bot 做完了，把要念的话交给它（没有就不汇报）
        self.device_key = device_key  # 后台盯着的任务按 (设备, Bot) 登记：会话关了再开，新会话也能接上它
        self._sub: Optional[Subscription] = None
        self._interrupted = False
        self._started = False
        self._in_turn = False

    @property
    def watch_key(self):
        return (self.device_key, self.config.bot_id)

    @property
    def session_id(self) -> Optional[str]:
        return self.config.bot_id

    @property
    def alive(self) -> bool:
        return self._started

    def _log(self, level: str, message: str) -> None:
        if self.logger:
            getattr(self.logger, level)(f"[{self.config.bot_name or self.config.bot_id[:8]}] {message}")

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        try:
            await self.hub.client.ensure_box()
        except GrokBotError as e:
            raise AgentSessionError(e.kind, e.detail)
        self._sub = self.hub.subscribe(self.config.bot_id)
        self._started = True

    async def close(self) -> None:
        """结束会话（设备断开、空闲超时）。正在进行的一轮当作被打断：Bot 在云端继续做，转后台盯着，做完再汇报。
        后台盯着的任务不受影响，它有自己的订阅。"""
        self._started = False
        if self._in_turn:
            self._interrupted = True
            if self._sub is not None:
                self._sub.queue.put_nowait(("interrupt", {}))
        sub, self._sub = self._sub, None
        if sub is not None:
            sub.close()

    async def interrupt(self) -> None:
        self._interrupted = True
        if self._sub is not None:
            self._sub.queue.put_nowait(("interrupt", {}))

    # ---------- 轮询与跟踪 ----------

    async def _poll(self, tracker: Tracker) -> List[str]:
        client = self.hub.client
        bot_id = self.config.bot_id
        agents = await client.list_agents()
        agent = next((a for a in agents if a.get("id") == bot_id), None)
        if agent is None:
            raise AgentSessionError(ErrorKind.UNKNOWN, f"云端没有这个 Bot：{bot_id}")
        texts = tracker.agent(agent)
        # 每次都拉尾部：newestEntryId 的含义没有文档，实测过一次它没变、尾部却有新条目的情况，不能只靠它省请求
        entries = await client.transcript_tail(bot_id, self.config.tail_limit)
        self._log("debug", f"{tracker._since()} 轮询：busy={agent_busy(agent)} newest={agent.get('newestEntryId')} "
                           f"尾部 {len(entries)} 条，已见 {len(tracker.seen)} 条")
        for e in entries:
            texts.extend(tracker.entry(e))
        return texts

    async def _follow(self, tracker: Tracker, queue: asyncio.Queue, stop: Callable[[], bool],
                      give_up: bool = True, deadline: Optional[float] = None) -> AsyncIterator[AgentEvent]:
        """跟着 Bot 的动静产出事件，直到这轮结束（产出 Done）或 stop() 为真、到了 deadline（都不产出 Done）。
        give_up：一直没回话时到 no_reply_timeout 就结束（设备在等）；后台盯着时不用，等到 deadline。"""
        cfg = self.config
        bot_id = cfg.bot_id
        next_poll = tracker.t0 + cfg.poll_interval
        while True:
            now = time.monotonic()
            if stop() or (deadline is not None and now >= deadline):
                return
            fin = tracker.finished(now)
            if fin == "done":
                self._log("info", f"{tracker._since()} 本轮结束，共 {tracker.spoken} 条回复")
                yield Done(session_id=bot_id)
                return
            if fin == "waiting":
                # Bot 在等用户去 App 里处理：这一轮到此为止，不管它的忙闲标志
                yield Done(session_id=bot_id)
                return
            if give_up and tracker.no_reply(now):
                self._log("warning", f"消息送到了，Bot {'忙过后' if tracker.was_busy else ''}空闲 "
                                     f"{int(cfg.no_reply_timeout)} 秒还没有回话，设备这边先结束")
                yield TextDelta(NO_REPLY_NOTICE)
                yield Done(session_id=bot_id)
                return

            waits = [next_poll - now]
            if deadline is not None:
                waits.append(deadline - now)
            if tracker.idle_pending:
                waits.append(tracker.idle_since + cfg.done_delay - now)
            if not tracker.echoed and not tracker.recovered:
                waits.append(tracker.t0 + cfg.echo_timeout - now)
            try:
                channel, payload = await asyncio.wait_for(queue.get(), timeout=max(0.05, min(waits)))
            except asyncio.TimeoutError:
                channel, payload = None, {}

            if channel == "transcript":
                if payload.get("type") in ("appended", "updated") and isinstance(payload.get("entry"), dict):
                    for t in tracker.entry(payload["entry"]):
                        yield TextDelta(t)
            elif channel == "agent-upserted":
                for t in tracker.agent(payload.get("agent") or {}):
                    yield TextDelta(t)
                yield Activity("agent", tools_running=1)
            elif channel == "hub-connected":
                if payload.get("generation") != tracker.generation:
                    # 事件流重连过，中间的事件可能漏了：补拉一次
                    tracker.generation = payload.get("generation")
                    for t in await self._poll(tracker):
                        yield TextDelta(t)
                    next_poll = time.monotonic() + cfg.poll_interval
            elif channel == "interrupt":
                continue

            now = time.monotonic()
            if not tracker.echoed and not tracker.recovered and now - tracker.t0 >= cfg.echo_timeout:
                # 事件流可能"假活"：连接还在，但什么都不推。重连并补拉
                tracker.recovered = True
                self._log("warning", f"发送 {int(cfg.echo_timeout)} 秒后事件流里没有回显，重连事件流并补拉对话")
                self.hub.reconnect()
                for t in await self._poll(tracker):
                    yield TextDelta(t)
                next_poll = time.monotonic() + cfg.poll_interval
            elif now >= next_poll:
                for t in await self._poll(tracker):
                    yield TextDelta(t)
                next_poll = time.monotonic() + cfg.poll_interval
                yield Activity("poll", tools_running=1)

    # ---------- 这一轮结束后在后台盯着 ----------

    def _start_watch(self, tracker: Tracker, texts: List[str] = ()) -> None:
        """这一轮在设备这边结束了（不管怎么结束的），留一个后台任务接着盯，之后到的回复交给 on_report。
        texts 是设备不听了之后这一轮才收到的回复（没有念过），一并汇报。
        Bot 已经做完（回过话、空闲了）只再盯 linger_max，接住补发的；还没拿到回复就盯 watch_max。"""
        if self.on_report is None:
            return
        replies = [t for t in texts if t not in NOTICES]
        if tracker.waiting and not replies:
            return  # 在等用户去 App 里处理，已经提示过；之后的回复用户在 App 里看
        linger = tracker.finished(time.monotonic()) == "done"
        tracker.next_batch(replied=bool(replies))  # 已经念过的不算：要等新的回复（或手上没念的）才汇报
        watch = AfterWatch(self, tracker, linger=linger)
        watch.texts = list(texts)
        old = _watches.pop(self.watch_key, None)
        if old is not None:
            old.take()  # 同一设备同一 Bot 只盯一个（上一个的话已经在这一轮念过或进了箱子）
        _watches[self.watch_key] = watch
        watch.start()

    def take_watch(self) -> str:
        """新的一轮开始前：停掉这台设备对这个 Bot 的后台盯着任务（哪怕是上一个会话留下的），
        把它已经收到、还没汇报的话拿回来先念。"""
        watch = _watches.pop(self.watch_key, None)
        return watch.take() if watch is not None else ""

    # ---------- 一轮对话 ----------

    async def send(self, text: str) -> AsyncIterator[AgentEvent]:
        if not self._started or self._sub is None:
            yield Error(ErrorKind.PROCESS_EXITED, "session not started", voice=voice_for(ErrorKind.PROCESS_EXITED))
            return
        self._interrupted = False
        # 从这里起就算在一轮里：发送前后（读尾部、sendPrompt 还没返回）设备断开也当打断，话照样发，转后台盯着
        self._in_turn = True
        try:
            async for ev in self._send(text):
                yield ev
        finally:
            self._in_turn = False

    async def _send(self, text: str) -> AsyncIterator[AgentEvent]:
        cfg = self.config
        client = self.hub.client
        bot_id = cfg.bot_id
        queue = self._sub.queue
        nonce = str(uuid.uuid4())
        sent_text = f"{text}（{cfg.voice_hint}）" if cfg.voice_hint else text
        seen: Dict[str, str] = {}

        # 上一件事被打断后它又回了话，先念了再发新的
        prelude = self.take_watch()
        if prelude:
            yield TextDelta(prelude)

        # 清掉上一轮残留的事件
        while not queue.empty():
            queue.get_nowait()

        # 发送前记下对话里已有的条目，之后只念新的。读不到也照常发：之后用回显的 seq 分新旧
        have_baseline = True
        try:
            for e in await client.transcript_tail(bot_id, cfg.tail_limit):
                if e.get("id"):
                    seen[str(e["id"])] = ""
        except GrokBotError as e:
            if e.kind == ErrorKind.AUTH:
                yield Error(e.kind, e.detail, voice=voice_for(e.kind))
                return
            have_baseline = False
            self._log("warning", f"读对话尾部失败，改用回显序号分新旧：{e.detail}")

        if not self.hub.connected and not await self.hub.wait_connected(cfg.connect_wait):
            self._log("warning", f"事件流 {cfg.connect_wait} 秒内没连上，先发送，靠补拉和轮询兜底")

        tracker = Tracker(cfg, nonce, sent_text, seen, self.hub.generation, self._log)
        tracker.have_baseline = have_baseline
        try:
            res = await client.send_prompt(bot_id, sent_text, nonce)
            self._log("info", f"{tracker._since()} sendPrompt 返回 {json.dumps(res, ensure_ascii=False)[:120]}")
        except GrokBotError as e:
            if e.kind == ErrorKind.AUTH:
                yield Error(e.kind, e.detail, voice=voice_for(e.kind))
                return
            # 结果不明确：查送达记录。只有查不到记录才是真的没送到（也只有这时才能用同一个 nonce 重发，这里不重发）
            try:
                status = await client.acceptance_status(bot_id, nonce)
            except GrokBotError as e2:
                yield Error(ErrorKind.SERVER, f"sendPrompt 失败：{e.detail}；查送达状态也失败：{e2.detail}",
                            voice=voice_for(ErrorKind.SERVER))
                return
            self._log("warning", f"sendPrompt 失败：{e.detail}；送达状态 {json.dumps(status, ensure_ascii=False)[:200]}")
            if "not-found" in json.dumps(status).lower() or "not_found" in json.dumps(status).lower():
                yield Error(ErrorKind.SERVER, f"消息没有送到：{e.detail}", voice="Grok Bot 没收到这句话，再说一次试试。")
                return

        yield Activity("sent", tools_running=1)
        held: List[str] = []  # 设备不听了（打断、断开）之后才到的回复，例如断开时正在补读：不念，留着汇报
        try:
            finished = False
            async for ev in self._follow(tracker, queue, lambda: self._interrupted):
                if self._interrupted and isinstance(ev, TextDelta):
                    held.append(ev.text)
                    continue
                finished = finished or isinstance(ev, Done)
                yield ev
            if self._interrupted and not finished:
                self._log("info", "本轮被打断，停止播报（云端继续做，做完再汇报）")
            # 打断、断开、没回话、做完了都一样留人盯着：Bot 想得久时会中途显示空闲，之后才发回复
            self._start_watch(tracker, held)
            if self._interrupted and not finished:
                yield Done(session_id=bot_id, interrupted=True)
        except asyncio.CancelledError:
            # 打断时正卡在慢请求里，被上层强制取消：一样转后台盯着
            self._start_watch(tracker, held)
            raise
        except (AgentSessionError, GrokBotError) as e:
            # 话已经送到了，等回复时出错（轮询失败等）：Bot 照样在做，留人盯着（盯的时候再出错就停）
            self._start_watch(tracker, held)
            yield Error(e.kind, e.detail, voice=voice_for(e.kind))


    async def hand_off(self, text: str) -> None:
        """设备已经不听了（启动期间被打断或断开），这句话照样交给 Bot：发出去后不等回复，直接转后台盯着，
        做完的回复交给 on_report。发送前拿回来的上一件事的回复也交给 on_report，免得丢。"""
        async for ev in self.send(text):
            if isinstance(ev, Activity) and ev.label == "sent":
                await self.interrupt()  # 发出去了：下面直接转后台盯着
            elif isinstance(ev, TextDelta) and ev.text.strip() and self.on_report is not None:
                self.on_report(ev.text)
            elif isinstance(ev, Error):
                self._log("warning", f"交出去的话没发成：{ev.detail[:200]}")


# 正在后台盯着的任务：(设备, Bot id) -> AfterWatch。asyncio 对任务只有弱引用，会话被关掉、对象被回收后任务会被
# 当垃圾清掉，这里保住它们；按键登记是为了会话关了再开（设备断连重连）时，新会话也能接上旧的
_watches: Dict[tuple, "AfterWatch"] = {}
NOTICES = (WAITING_NOTICE, NO_REPLY_NOTICE)


class AfterWatch:
    """这一轮结束后继续盯着这个 Bot：用自己的订阅收后续回复，每做完一批（或它等用户）交给 on_report。

    汇报完一批后再盯 linger_max：Bot 可能先回一句"我去查一下"，过几分钟才发结果。"""

    def __init__(self, session: GrokBotSession, tracker: Tracker, linger: bool = False):
        self.session = session
        self.tracker = tracker
        self.linger = linger  # 这一轮已经做完，只是再盯一会儿
        self.texts: List[str] = []
        self.task: Optional[asyncio.Task] = None
        self.sub: Optional[Subscription] = None
        self.done = False

    def start(self) -> None:
        cfg = self.session.config
        self.sub = self.session.hub.subscribe(cfg.bot_id)
        self.tracker.generation = self.session.hub.generation
        self.task = asyncio.ensure_future(self._run())
        if self.linger:
            self.session._log("info", f"本轮做完了，再盯 {int(cfg.linger_max // 60)} 分钟，接住它补发的")
        else:
            self.session._log("info", f"还没拿到回复，后台盯着它，最多 {int(cfg.watch_max // 60)} 分钟")

    async def _run(self) -> None:
        cfg = self.session.config
        deadline = time.monotonic() + (cfg.linger_max if self.linger else cfg.watch_max)
        try:
            while not self.done:
                ended = False
                async for ev in self.session._follow(self.tracker, self.sub.queue, lambda: self.done,
                                                     give_up=False, deadline=deadline):
                    if isinstance(ev, TextDelta):
                        self.texts.append(ev.text)
                    elif isinstance(ev, Done):
                        ended = True
                if not ended:
                    break  # 盯到时间了
                self._report()
                if self.tracker.waiting:
                    self.done = True  # 在等用户去 App 里处理（已汇报）：后面的事用户在 App 里看
                    break
                self.tracker.next_batch()
                deadline = time.monotonic() + cfg.linger_max  # 做完一批了：之后只再盯 linger_max
        except asyncio.CancelledError:
            raise
        except (AgentSessionError, GrokBotError) as e:
            self.session._log("warning", f"后台盯着时出错，停止：{e.detail}")
        except Exception as e:  # noqa: BLE001 —— 后台任务，意外也要把已收到的话汇报出去
            self.session._log("error", f"后台盯着时意外出错，停止：{e!r}")
        finally:
            if self.sub is not None:
                self.sub.close()
            if _watches.get(self.session.watch_key) is self:
                _watches.pop(self.session.watch_key, None)
        self._report()  # 盯到时间时手上还有没汇报的（Bot 一直没空闲下来），也交出去
        if not self.done:
            self.done = True
            self.session._log("info", "后台盯着结束")

    def _report(self) -> None:
        """把手上的话交给 on_report（交出去就清空，下一轮不再重复念）。"""
        if self.done:
            return
        name = self.session.config.bot_name or "Grok Bot"
        texts, self.texts = [t for t in self.texts if t not in NOTICES], []
        parts = [f"{name}做完了：" + "".join(texts)] if texts else []
        if self.tracker.waiting:
            parts.append(f"{name}在等你去 App 里处理。")
        if not parts:
            self.session._log("info", "后台盯着：这一批没有新回复，不汇报")
            return
        text = "".join(parts)
        self.session._log("info", f"待汇报：{text[:80]}")
        if self.session.on_report is not None:
            self.session.on_report(text)

    def take(self) -> str:
        """新的一轮要开始了：停下，把已收到、还没汇报的话拿走（之后的回复新的一轮自己会念）。"""
        if self.done:
            return ""
        self.done = True
        if self.task is not None and not self.task.done():
            self.task.cancel()
        texts, self.texts = [t for t in self.texts if t not in NOTICES], []
        if not texts:
            return ""  # 汇报过的已经进了待汇报箱，那边会念
        name = self.session.config.bot_name or "Grok Bot"
        return f"{name}刚才回了：" + "".join(texts)
