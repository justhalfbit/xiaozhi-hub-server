"""Independent task context, current evidence and endpoint guards. No paid API calls."""
import asyncio
from types import SimpleNamespace
import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web
from core import scheduled_tasks as st
from core.admin.admin_handler import AdminHandler
from plugins_func.functions import web_search as ws
pytestmark = pytest.mark.asyncio

@pytest.fixture
def cfg(monkeypatch):
    from core.agent import llm_router
    monkeypatch.setattr(llm_router, 'store', lambda: SimpleNamespace(get=lambda _: None))
    return {'selected_module': {'LLM': 'Test'}, 'LLM': {'Test': {
        'type': 'openai', 'base_url': 'https://model.example/v1', 'model_name': 'test', 'api_key': 'fake-key'}}}

async def test_weather_is_fresh_and_context_is_independent(cfg, monkeypatch):
    from plugins_func.functions import get_weather as gw
    calls, prompts = [], []
    async def weather(conn, location, fresh=False):
        calls.append((location, fresh))
        return SimpleNamespace(result='地点：杭州\n当前天气：晴天')
    monkeypatch.setattr(gw, 'get_weather', weather)
    async def ask(entry, messages):
        prompts.append(messages)
        return '{"location":"杭州","query":""}' if len(prompts) == 1 else '杭州今天晴天。'
    result = await st.execute(cfg, {'instruction': '播报当天杭州天气', 'tools': ['get_weather']}, complete_fn=ask)
    assert result['text'] == '杭州今天晴天。' and calls == [('杭州', True)]
    assert all([m['role'] for m in p] == ['system', 'user'] for p in prompts)
    assert '当前天气' in prompts[-1][1]['content']

async def test_search_uses_runtime_date_and_keeps_sources(cfg, monkeypatch):
    from plugins_func.functions import web_search as ws
    queries = []
    async def search(settings, query):
        queries.append(query)
        return ws.SearchResult(summary='未找到已核实的优惠', results=[{
            'title': '官方说明', 'url': 'https://openai.com/test', 'page_age': '', 'snippet': '条件说明'}])
    monkeypatch.setattr(ws, 'search', search)
    answers = iter(['{"query":"某服务 当前优惠","location":""}', '未找到已核实的优惠。'])
    async def ask(*args): return next(answers)
    result = await st.execute(cfg, {'instruction': '圣诞节查优惠', 'tools': ['web_search']}, complete_fn=ask)
    assert '执行日期' in queries[0] and '有效期' in queries[0]
    assert result['sources'][0]['url'] == 'https://openai.com/test'

async def test_search_failure_is_not_no_discount(cfg, monkeypatch):
    from plugins_func.functions import web_search as ws
    async def search(*args): raise ws.SearchError('failed')
    monkeypatch.setattr(ws, 'search', search)
    calls = []
    async def ask(*args): calls.append(1); return '{"query":"优惠","location":""}'
    with pytest.raises(st.TaskError, match='不能据此判断没有活动'):
        await st.execute(cfg, {'instruction': '查最新优惠', 'tools': ['web_search']}, complete_fn=ask)
    assert len(calls) == 1

async def test_generation_uses_one_model_call(cfg):
    prompts = []
    async def ask(entry, messages): prompts.append(messages); return '成语水滴石穿。'
    result = await st.execute(cfg, {'instruction': '讲一个成语', 'tools': []}, complete_fn=ask)
    assert result['sources'] == [] and len(prompts) == 1

@pytest.mark.parametrize('tools', [['shell'], ['self.task.create'], 'web_search', [None]])
async def test_no_arbitrary_tools(cfg, tools):
    with pytest.raises(ValueError): st.validate({'instruction': 'test', 'tools': tools})

async def test_unsupported_model_does_not_silently_switch(cfg):
    cfg['LLM']['Test']['type'] = 'claude_cli'
    with pytest.raises(st.TaskError, match='API 模型'): st.model_config(cfg, '')

async def test_endpoint_local_only_busy_and_timeout_recovery(cfg, monkeypatch, tmp_path):
    handler = AdminHandler(cfg, config_path=tmp_path / '.config.yaml')
    app = web.Application(); handler.register(app)
    client = TestClient(TestServer(app)); await client.start_server()
    try:
        body = {'instruction': '讲一个成语', 'tools': [], 'device':'qa', 'run_id':'test'}
        assert (await client.post('/kiosk/tasks/run', json=body, headers={'X-Forwarded-For': '127.0.0.1'})).status == 403
        handler._scheduled_busy = True
        assert (await client.post('/kiosk/tasks/run', json=body)).status == 409
        handler._scheduled_busy = False
        async def timeout(*args): raise asyncio.TimeoutError()
        monkeypatch.setattr(st, 'execute', timeout)
        assert (await client.post('/kiosk/tasks/run', json=body)).status == 504
        assert not handler._scheduled_busy
        async def good(*args): return {'ok': True, 'text': '成语故事'}
        monkeypatch.setattr(st, 'execute', good)
        assert (await (await client.post('/kiosk/tasks/run', json=body)).json())['text'] == '成语故事'
    finally:
        await client.close()

async def test_cancel_only_matching_execution_closes_gateway_work(cfg, monkeypatch, tmp_path):
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def long_work(*args):
        started.set()
        try: await asyncio.Event().wait()
        finally: cancelled.set()
    monkeypatch.setattr(st, 'execute', long_work)
    handler = AdminHandler(cfg, config_path=tmp_path / '.config.yaml')
    app = web.Application(); handler.register(app)
    client = TestClient(TestServer(app)); await client.start_server()
    try:
        work = asyncio.ensure_future(client.post('/kiosk/tasks/run', json={
            'instruction':'生成内容','tools':[],'device':'qa','run_id':'one'}))
        await started.wait()
        wrong = await client.post('/kiosk/tasks/cancel', json={'device':'qa','run_id':'old'})
        assert not (await wrong.json())['stopped'] and not cancelled.is_set()
        right = await client.post('/kiosk/tasks/cancel', json={'device':'qa','run_id':'one'})
        assert (await right.json())['stopped']
        assert (await work).status == 409
        assert cancelled.is_set() and not handler._scheduled_busy
        stale = await client.post('/kiosk/tasks/run', json={'instruction':'生成内容','tools':[],'device':'qa','run_id':'old'})
        assert stale.status == 409
    finally:
        await client.close()

@pytest.mark.parametrize('base,expected', [
    ('https://api.deepseek.com/v1', {'thinking': {'type':'disabled'}}),
    ('https://dashscope.aliyuncs.com/compatible-mode/v1', {'enable_thinking':False}),
    ('https://deepseek.com.example.org/v1', {}),
])
async def test_background_api_uses_same_reasoning_defaults_as_voice(cfg, monkeypatch, base, expected):
    import httpx
    from httpx._client import AsyncClient
    seen=[]
    def handler(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':'结果'},'finish_reason':'stop'}]})
    monkeypatch.setattr(st.httpx,'AsyncClient',lambda **kw:AsyncClient(transport=httpx.MockTransport(handler),**kw))
    entry={**cfg['LLM']['Test'],'base_url':base}
    assert await st.complete(entry,[{'role':'user','content':'任务'}])=='结果'
    assert {k:seen[0][k] for k in expected}==expected
    if not expected: assert 'thinking' not in seen[0] and 'enable_thinking' not in seen[0]

async def test_explicit_model_options_preserved_but_cannot_replace_task_context(cfg, monkeypatch):
    import httpx,json
    from httpx._client import AsyncClient
    seen=[]
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':None},'finish_reason':'length'}]})
    monkeypatch.setattr(st.httpx,'AsyncClient',lambda **kw:AsyncClient(transport=httpx.MockTransport(handler),**kw))
    entry={**cfg['LLM']['Test'],'base_url':'https://api.deepseek.com/v1',
           'extra_body':{'thinking':{'type':'enabled'},'messages':[], 'stream':True}}
    with pytest.raises(st.TaskError,match='长度限制'):
        await st.complete(entry,[{'role':'user','content':'任务'}])
    assert seen[0]['thinking']=={'type':'enabled'}
    assert seen[0]['messages'] and seen[0]['stream'] is False

async def test_long_summary_gets_one_bounded_compression_without_new_search(cfg):
    calls=[]
    async def ask(entry,messages):
        calls.append(messages)
        return '长'*250 if len(calls)==1 else '有效期和地区条件明确，未找到已核实的通用优惠。'
    result=await st.execute(cfg,{'instruction':'生成一段简报','tools':[]},complete_fn=ask)
    assert len(calls)==2 and len(result['text'])<=200
    assert calls[1][-2]['role']=='assistant' and len(calls[1][-2]['content'])==250

async def test_compression_is_not_an_unbounded_retry(cfg):
    calls=[]
    async def ask(*args): calls.append(1);return '长'*250
    result=await st.execute(cfg,{'instruction':'生成内容','tools':[]},complete_fn=ask)
    assert result['text']=='长'*250 and len(calls)==2


async def test_extreme_output_is_still_bounded(cfg):
    async def ask(*args): return '长'*1300
    with pytest.raises(st.TaskError,match='过长'):
        await st.execute(cfg,{'instruction':'生成内容','tools':[]},complete_fn=ask)


async def test_weather_failure_never_reaches_summary_or_fabricates_forecast(cfg, monkeypatch):
    from plugins_func.functions import get_weather as gw
    async def weather(*args, **kwargs):
        return SimpleNamespace(result="没查到天气：服务暂时连不上")
    monkeypatch.setattr(gw, "get_weather", weather)
    prompts = []
    async def ask(entry, messages):
        prompts.append(messages)
        return '{"location":"","query":""}'
    with pytest.raises(st.TaskError, match="天气查询失败"):
        await st.execute(cfg, {"instruction": "播报当天天气", "tools": ["get_weather"]}, complete_fn=ask)
    assert len(prompts) == 1
