"""每一轮对话的代次：打断或开始新的一轮就加一，旧代次的识别结果、模型输出一律丢掉。

为什么要有：识别在线程里跑（板子上 1～2 秒）、模型在线程里流式返回，打断管不到它们。
原来靠 client_abort 一个开关，可新的一轮一开始就把它清回 False，晚到的旧结果就混进来了：
识别中打断，旧那句话识别完照样拿去问模型；思考中打断，旧回答的字和声音还可能在新一轮里播出来。

做法：
- 设备打断（abort）、开始新的一轮（startToChat）都把 conn.turn_gen 加一；
- 说完开始识别时记下当时的代次，识别完代次变了就丢掉，不问模型、不发 stt；
- 每轮 chat() 绑定开始这一轮时的代次（bind），之后读模型流、往合成里塞字、写对话前都查 stale()。
"""

import threading

KEEP = 8

_lock = threading.Lock()


def current(conn) -> int:
    return getattr(conn, "turn_gen", 0)


def bump(conn) -> int:
    """打断或开始新的一轮：之前的代次全部作废。"""
    with _lock:
        conn.turn_gen = current(conn) + 1
        return conn.turn_gen


def bind(conn, sentence_id, gen) -> None:
    """chat() 这一轮（sentence_id）属于哪一代。"""
    with _lock:
        sids = getattr(conn, "turn_sids", None)
        if sids is None:
            sids = conn.turn_sids = {}
        sids[sentence_id] = gen
        for old in list(sids)[:-KEEP]:
            del sids[old]


def stale(conn, gen=None, sentence_id=None) -> bool:
    """这一代（或这一轮 chat 所属的代）已经过期了吗。"""
    if sentence_id is not None:
        gen = getattr(conn, "turn_sids", {}).get(sentence_id, gen)
    return gen is not None and gen != current(conn)
