"""查天气：选数据源（auto / open-meteo / qweather）、Open-Meteo 挑地点和报告格式、本地位置、IP 和默认城市、缓存、出错。

不联网：请求接到 httpx.MockTransport 上。地名搜索的样例取自实际请求 geocoding-api.open-meteo.com 的结果（只留用得到的字段）。
"""

from types import SimpleNamespace

import httpx
import pytest

from plugins_func.functions import get_weather as gw
from plugins_func.register import Action


def place(name, admin1, cc, lat, lon, code=None, pop=None, country="中国"):
    return {"name": name, "admin1": admin1, "country_code": cc, "country": country,
            "latitude": lat, "longitude": lon, "feature_code": code, "population": pop}


GEO = {
    "长沙": [place("长沙", "重庆市", "CN", 31.0, 108.3, "PPLA4", 47170)],
    "Changsha": [place("长沙市", "湖南", "CN", 28.2, 112.97, "PPLA", 3093980)],
    "Suzhou": [place("宿州市", "安徽", "CN", 33.64, 116.98, "PPLA2", 5300000),
               place("苏州", "江苏", "CN", 31.31, 120.6, "PPLA2", 6715559)],
    "Dali": [place("大理市", "云南", "CN", 25.6, 100.27, "PPLA2", 650000),
             place("大连市", "辽宁", "CN", 38.91, 121.6, "PPLA2", 7450785)],
    "大理": [place("大理", "四川", "CN", 29.0, 103.0, "PPL", None)],
    "北京": [place("北京", "北京市", "CN", 39.91, 116.4, "PPLC", 18960744)],
    "广州": [place("广州", "广东", "CN", 23.12, 113.25, "PPLA", 18676605)],
    "Tokyo": [place("東京", "东京都", "JP", 35.69, 139.69, "PPLC", 9733276, "日本"),
              place("东京", "江苏", "CN", 32.0, 119.0, "PPL", None)],
}

FORECAST = {
    "current": {"time": "2026-11-07T00:15", "temperature_2m": 8.4, "apparent_temperature": 5.1,
                "relative_humidity_2m": 41, "weather_code": 0, "wind_speed_10m": 14.0,
                "wind_direction_10m": 315, "precipitation": 0},
    "daily": {"time": ["2026-11-07", "2026-11-08", "2026-11-09"],
              "weather_code": [0, 61, 3], "temperature_2m_max": [12.3, 9.0, 10.6],
              "temperature_2m_min": [1.6, 3.2, 2.0], "precipitation_probability_max": [0, 80, 10]},
}

# 和风的城市查询和预报页（只留 parse_weather_info 用到的结构）
QW_CITY = {"code": "200", "location": [{"name": "杭州", "fxLink": "https://www.qweather.com/weather/hangzhou-101210101.html"}]}
QW_PAGE = """<html><body>
<h1 class="c-submenu__location">杭州</h1>
<div class="c-city-weather-current"><div class="current-abstract">晴 18°</div>
<div class="current-basic"><div class="current-basic___item"><p>3级</p><p>东风</p></div></div></div>
<div class="city-forecast-tabs__row"><div class="date-bg"><span class="date">今天</span><img class="icon" src="/img/100.svg"></div>
<div class="tmp-cont"><span class="temp">20°</span><span class="temp">12°</span></div></div>
</body></html>"""


@pytest.fixture
def api(monkeypatch):
    """把插件发出的请求接到假的接口上，记下请求；state["down"] = True 时模拟网络不通。"""
    calls, state = [], {"down": False}

    def handler(request):
        calls.append(request)
        if state["down"]:
            raise httpx.ConnectTimeout("timed out", request=request)
        url, q = request.url, request.url.params
        if url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": GEO.get(q["name"], [])} if q["name"] in GEO else {})
        if url.host == "api.open-meteo.com":
            return httpx.Response(200, json=FORECAST)
        if url.path == "/geo/v2/city/lookup":
            return httpx.Response(200, json=QW_CITY)
        if url.host == "www.qweather.com":
            return httpx.Response(200, text=QW_PAGE)
        return httpx.Response(404)

    from httpx._client import AsyncClient as real  # tests/conftest.py 把 httpx.AsyncClient 换成了连不上的桩
    monkeypatch.setattr(gw.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return SimpleNamespace(calls=calls, state=state, hosts=lambda: [c.url.host for c in calls])


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    """每个测试一份干净的缓存，记下 ttl；按 IP 查城市不联网（默认查不到）。"""
    from core.utils.cache.manager import cache_manager

    store = {}
    monkeypatch.setattr(cache_manager, "get", lambda kind, key, namespace="": store.get((kind, key), (None,))[0])
    monkeypatch.setattr(cache_manager, "set",
                        lambda kind, key, value, ttl=None, namespace="": store.__setitem__((kind, key), (value, ttl)))
    monkeypatch.setattr(gw, "get_ip_info", lambda ip, logger: {})
    return store


LOCAL = {"name": "北京", "latitude": 39.9075, "longitude": 116.3972, "aliases": ["Beijing"]}


def conn(ip=None, **cfg):
    return SimpleNamespace(config={"plugins": {"get_weather": cfg}}, client_ip=ip)


# ---------- 选数据源 ----------

def test_provider_auto_uses_qweather_only_with_an_own_key():
    assert gw.provider_of({}) == "open-meteo"
    assert gw.provider_of({"provider": "auto", "api_key": ""}) == "open-meteo"
    assert gw.provider_of({"api_key": gw.SHARED_QWEATHER_KEY}) == "open-meteo"   # 失效的共用 Key 当成没填
    assert gw.provider_of({"api_key": "你的和风天气API密钥"}) == "open-meteo"
    assert gw.provider_of({"api_key": "my-own-key"}) == "qweather"                # 以前配好和风的不用改
    assert gw.provider_of({"provider": "open-meteo", "api_key": "my-own-key"}) == "open-meteo"
    assert gw.provider_of({"provider": "Open_Meteo"}) == "open-meteo"
    assert gw.provider_of({"provider": "qweather"}) == "qweather"


def test_function_name_and_parameters_stay_compatible():
    fn = gw.GET_WEATHER_FUNCTION_DESC["function"]
    assert fn["name"] == "get_weather" and fn["parameters"]["required"] == ["lang"]
    assert {"location", "lang"} <= set(fn["parameters"]["properties"])


# ---------- Open-Meteo：挑地点 ----------

def test_pinyin_with_province_picks_the_right_namesake():
    assert gw.pick_place(GEO["Suzhou"], "Suzhou", "江苏")["name"] == "苏州"
    assert gw.pick_place(GEO["Suzhou"], "Suzhou", "江苏省")["name"] == "苏州"
    assert gw.pick_place(GEO["Dali"], "Dali", "云南")["name"] == "大理市"
    assert gw.pick_place(GEO["Suzhou"], "Suzhou")["name"] == "宿州市"   # 不给省份就按搜索结果的顺序


def test_country_code():
    assert gw.pick_place(GEO["Tokyo"], "Tokyo", country_code="jp")["country_code"] == "JP"


def test_chinese_names_only_count_when_they_are_real_cities():
    assert gw.pick_place(GEO["长沙"], "长沙") is None   # 重庆的一个镇
    assert gw.pick_place(GEO["大理"], "大理") is None   # 四川的一个村
    assert gw.pick_place(GEO["北京"], "北京市")["name"] == "北京"


def test_wind_and_numbers():
    assert gw._wind(14.0, 315) == "西北风3级"
    assert gw._wind(0.5, 90) == "无风"
    assert gw._wind(30, 0) == "北风5级"
    assert gw._num(-0.2) == "0"


# ---------- Open-Meteo：查天气 ----------

async def test_report(api):
    r = await gw.get_weather(conn(local=LOCAL))
    assert r.action == Action.REQLLM
    assert "当前天气：晴，气温 8°C，体感 5°C，湿度 41%，西北风3级" in r.result
    assert "今天 11月7日 周六：晴，气温 2~12°C，降水概率 0%" in r.result
    assert "明天 11月8日 周日：小雨，气温 3~9°C，降水概率 80%" in r.result
    assert "后天 11月9日 周一：阴" in r.result


async def test_local_needs_no_place_search(api, cache):
    for loc in (None, "北京", "北京市", "Beijing"):
        api.calls.clear()
        cache.clear()
        r = await gw.get_weather(conn(local=LOCAL), location=loc)
        assert api.hosts() == ["api.open-meteo.com"], loc
        assert "您查询的位置是：北京" in r.result
        assert api.calls[0].url.params["latitude"] == "39.9075"


async def test_ip_city_counts_as_local(api, cache):
    """prompt_manager 拿按 IP 查到的城市当地点调本函数：配了 local 时它也算本地。"""
    from core.utils.cache.manager import CacheType
    cache[(CacheType.LOCATION, "1.2.3.4")] = ("北京市", None)
    await gw.get_weather(conn(ip="1.2.3.4", local={**LOCAL, "aliases": []}), location="北京市")
    assert api.hosts() == ["api.open-meteo.com"]


async def test_other_city(api):
    r = await gw.get_weather(conn(local=LOCAL), location="Changsha", province="湖南")
    assert "您查询的位置是：长沙市（湖南，中国）" in r.result
    assert api.calls[-1].url.params["latitude"] == "28.2000"


async def test_chinese_name_that_misleads_the_search_asks_for_pinyin(api):
    r = await gw.get_weather(conn(local=LOCAL), location="长沙")
    assert "没找到「长沙」" in r.result and "拼音" in r.result
    assert api.hosts() == ["geocoding-api.open-meteo.com"]   # 不给错的长沙查预报


async def test_without_local_uses_ip_city_then_default_location(api, monkeypatch):
    r = await gw.get_weather(conn(default_location="广州"))
    assert "广州（广东，中国）" in r.result
    monkeypatch.setattr(gw, "get_ip_info", lambda ip, logger: {"city": "北京市"})
    api.calls.clear()
    r = await gw.get_weather(conn(ip="1.2.3.4", default_location="广州"))
    assert "北京（北京市，中国）" in r.result
    assert api.calls[0].url.params["name"] == "北京"          # 去掉"市"再搜，「北京市」搜不到


async def test_nothing_configured_asks_for_a_city(api):
    r = await gw.get_weather(conn())
    assert "请告诉我要查哪个城市" in r.result and api.calls == []


async def test_network_failure(api):
    api.state["down"] = True
    r = await gw.get_weather(conn(local=LOCAL))
    assert r.result is None and "连不上" in r.response


async def test_cache_keeps_twenty_minutes(api, cache):
    await gw.get_weather(conn(local=LOCAL))
    r = await gw.get_weather(conn(local=LOCAL))
    assert len(api.calls) == 1 and "您查询的位置是：北京" in r.result
    assert [ttl for _, ttl in cache.values()] == [gw.OPEN_METEO_CACHE_SECONDS]


# ---------- 和风 ----------

async def test_qweather_with_own_key(api):
    r = await gw.get_weather(conn(api_key="my-own-key", api_host="abc.re.qweatherapi.com"),
                             location="Hangzhou", province="浙江")
    lookup = api.calls[0]
    assert lookup.url.host == "abc.re.qweatherapi.com"
    assert lookup.url.params["location"] == "Hangzhou" and lookup.url.params["adm"] == "浙江"
    assert lookup.url.params["key"] == "my-own-key"
    assert "您查询的位置是：杭州" in r.result and "当前天气: 晴 18°" in r.result
    assert "今天: 晴，气温 12°~20°" in r.result
    assert "open-meteo" not in " ".join(api.hosts())


async def test_qweather_local_uses_coordinates(api):
    await gw.get_weather(conn(provider="qweather", api_key="k", api_host="h.example", local=LOCAL))
    assert api.calls[0].url.params["location"] == "116.40,39.91"


async def test_qweather_failure_does_not_raise(api):
    api.state["down"] = True
    r = await gw.get_weather(conn(provider="qweather", api_key="k", api_host="h.example"), location="Hangzhou")
    assert r.result is None and "连不上" in r.response
