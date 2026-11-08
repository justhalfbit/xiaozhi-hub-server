"""查天气（Open-Meteo + Nominatim）：报告格式、地名查询和缓存、本地所在地的顺序、后台设置和测试按钮。"""

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from core.admin import catalog, plugin_settings
from plugins_func.functions import get_weather as gw

NOW = "2026-11-07T21:00"


def forecast(daily_codes=(3, 61, 71), rain=(0.2, 5.0, 3.0), hourly=None):
    hourly = hourly or {}
    times = [f"2026-11-07T{h:02d}:00" for h in range(24)] + [f"2026-11-08T{h:02d}:00" for h in range(24)]
    return {
        "current": {"time": NOW, "temperature_2m": 24.4, "relative_humidity_2m": 90, "apparent_temperature": 28.6,
                    "weather_code": 3, "wind_speed_10m": 14.0, "wind_direction_10m": 45},
        "hourly": {"time": times,
                   "precipitation_probability": [hourly.get(t, (0, 0, 3))[0] for t in times],
                   "precipitation": [hourly.get(t, (0, 0, 3))[1] for t in times],
                   "weather_code": [hourly.get(t, (0, 0, 3))[2] for t in times]},
        "daily": {"time": ["2026-11-07", "2026-11-08", "2026-11-09"][:len(daily_codes)],
                  "weather_code": list(daily_codes), "temperature_2m_max": [29.8, 29.6, 23.6],
                  "temperature_2m_min": [22.9, 21.1, 16.0], "precipitation_sum": list(rain),
                  "precipitation_probability_max": [31, 85, 73]},
    }


PLACE = gw.Place("深圳市（广东省）", 22.54, 114.05)


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    gw._places.clear()
    gw._reports.clear()
    monkeypatch.setattr(gw, "NOMINATIM_INTERVAL", 0)


@pytest.fixture
def fake_api(monkeypatch):
    """把插件发出的请求接到假的接口上：routes 按域名给回应。"""
    calls = []
    routes = {
        "nominatim.openstreetmap.org": lambda req: httpx.Response(200, json=[
            {"display_name": "天河区, 广州市, 广东省, 中国", "lat": "23.12", "lon": "113.36"}]),
        "geocoding-api.open-meteo.com": lambda req: httpx.Response(200, json={"results": []}),
        "api.open-meteo.com": lambda req: httpx.Response(200, json=forecast()),
        "air-quality-api.open-meteo.com": lambda req: httpx.Response(200, json={"current": {"pm2_5": 48.4}}),
    }

    def handler(request):
        calls.append(request)
        return routes[request.url.host](request)

    from httpx._client import AsyncClient as real  # tests/conftest.py 把 httpx.AsyncClient 换成了连不上的桩
    monkeypatch.setattr(gw.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return SimpleNamespace(calls=calls, routes=routes)


def hosts(calls):
    return [c.url.host for c in calls]


# ---------- 报告 ----------

def test_report_is_short_lines_with_day_names():
    text = gw.format_report(PLACE, forecast(), {"current": {"pm2_5": 48.4}})
    lines = text.splitlines()
    assert lines[0] == "地点：深圳市（广东省）"
    assert lines[1] == "现在（21:00）：阴，24°C，体感 29°C，湿度 90%，东北风 3 级"
    assert lines[2] == "空气质量：良（PM2.5 48）"
    assert lines[3] == "接下来 12 小时：基本不会下雨"
    # 0.2 毫米的「阴」不变；雨量够的小雨照说；雪照说
    assert lines[4] == "今天11月7日 周六：阴，23～30°C，降水概率 31%"
    assert lines[5] == "明天11月8日 周日：小雨，21～30°C，降水概率 85%"
    assert lines[6] == "后天11月9日 周一：小雪，16～24°C，降水概率 73%"


def test_tiny_drizzle_is_not_called_rain():
    text = gw.format_report(PLACE, forecast(daily_codes=(53, 55), rain=(0.3, 2.5)))
    assert "今天11月7日 周六：多云，可能有零星小雨" in text
    assert "明天11月8日 周日：小雨" in text  # 毛毛雨但雨量到了 1 毫米
    assert "空气质量" not in text  # 空气质量没查到就不写


def test_rain_window_across_midnight_and_snow():
    wet = {f"2026-11-07T{h}:00": (60, 0.5, 61) for h in (22, 23)}
    wet["2026-11-08T00:00"] = (80, 1.0, 61)
    text = gw.format_report(PLACE, forecast(hourly=wet))
    assert "接下来 12 小时：22 点到明天1 点可能下雨，降水概率最高 80%" in text

    snow = {"2026-11-07T23:00": (90, 0.4, 73)}
    assert "接下来 12 小时：23 点可能下雪，降水概率最高 90%" in gw.format_report(PLACE, forecast(hourly=snow))


def test_place_label_names_province_and_city():
    assert gw._label(["南山街道", "南山区", "深圳市", "广东省", "518000", "中国"]) == "南山街道（广东省深圳市南山区）"
    assert gw._label(["南山区", "鹤岗市", "黑龙江省", "154100", "中国"]) == "南山区（黑龙江省鹤岗市）"
    assert gw._label(["北京市", "中国"]) == "北京市"
    assert gw._label(["纽约", "纽约州", "美国"]) == "纽约（纽约州）"


def test_wind_levels():
    assert gw._wind(0.5, 0) == "无风"
    assert gw._wind(25, 180) == "南风 4 级"
    assert gw._wind(None, None) == ""


# ---------- 地名 ----------

def test_find_place_uses_nominatim_and_caches(fake_api):
    place = asyncio.run(gw.find_place("天河"))
    assert place == gw.Place("天河区（广东省广州市）", 23.12, 113.36)
    asyncio.run(gw.find_place("天河"))
    assert hosts(fake_api.calls) == ["nominatim.openstreetmap.org"]  # 第二次走缓存
    assert "xiaozhi-hub-server" in fake_api.calls[0].headers["user-agent"]
    assert fake_api.calls[0].url.params["accept-language"] == "zh-CN"
    assert fake_api.calls[0].url.params["featureType"] == "settlement"  # 只找居民点和行政区


def test_find_place_falls_back_to_open_meteo(fake_api):
    fake_api.routes["nominatim.openstreetmap.org"] = lambda req: httpx.Response(503)
    fake_api.routes["geocoding-api.open-meteo.com"] = lambda req: httpx.Response(200, json={"results": [
        {"name": "朝阳", "admin1": "重庆市", "latitude": 1, "longitude": 2, "population": 13524},
        {"name": "朝阳", "admin1": "辽宁", "latitude": 41.57, "longitude": 120.45, "population": 300000},
        {"name": "朝阳", "admin1": "山西", "latitude": 3, "longitude": 4}]})
    assert asyncio.run(gw.find_place("朝阳")) == gw.Place("朝阳（辽宁）", 41.57, 120.45)


def test_unknown_place_and_service_down(fake_api):
    fake_api.routes["nominatim.openstreetmap.org"] = lambda req: httpx.Response(200, json=[])
    with pytest.raises(gw.WeatherError, match="没找到「火星」"):
        asyncio.run(gw.find_place("火星"))
    count = len(fake_api.calls)
    with pytest.raises(gw.WeatherError):
        asyncio.run(gw.find_place("火星"))
    assert len(fake_api.calls) == count  # 查不到的也缓存一会儿

    fake_api.routes["nominatim.openstreetmap.org"] = lambda req: httpx.Response(500)
    fake_api.routes["geocoding-api.open-meteo.com"] = lambda req: httpx.Response(500)
    with pytest.raises(gw.WeatherError, match="连不上"):
        asyncio.run(gw.find_place("杭州"))


# ---------- 插件 ----------

def conn_with(location="", ip_city=None, monkeypatch=None):
    conn = SimpleNamespace(config={"plugins": {"get_weather": {"location": location, "default_location": "广州"}}},
                           client_ip="127.0.0.1", loop=None)
    if monkeypatch is not None:
        monkeypatch.setattr(gw, "_ip_city", lambda conn: ip_city or "")
    return conn


def test_local_query_order(monkeypatch):
    assert gw.local_query(conn_with("深圳南山区", "杭州市", monkeypatch)) == "深圳南山区"
    assert gw.local_query(conn_with("", "杭州市", monkeypatch)) == "杭州市"
    assert gw.local_query(conn_with("", None, monkeypatch)) == "广州"


def test_get_weather_tool(fake_api, monkeypatch):
    conn = conn_with("天河", monkeypatch=monkeypatch)
    r = asyncio.run(gw.get_weather(conn))
    assert r.result.startswith("地点：天河区（广东省广州市）")
    assert sorted(hosts(fake_api.calls)) == ["air-quality-api.open-meteo.com", "api.open-meteo.com",
                                             "nominatim.openstreetmap.org"]
    asyncio.run(gw.get_weather(conn, location="天河"))
    assert len(fake_api.calls) == 3  # 天气缓存 1 小时

    fake_api.routes["api.open-meteo.com"] = lambda req: httpx.Response(502)
    gw._reports.clear()
    r = asyncio.run(gw.get_weather(conn, location="天河"))
    assert r.result.startswith("没查到天气：天气服务暂时连不上")


def test_air_quality_failure_still_reports(fake_api, monkeypatch):
    fake_api.routes["air-quality-api.open-meteo.com"] = lambda req: httpx.Response(500)
    r = asyncio.run(gw.get_weather(conn_with(monkeypatch=monkeypatch), location="天河"))
    assert "地点：天河区（广东省广州市）" in r.result and "空气质量" not in r.result


# ---------- 提示词 ----------

# ---------- 后台 ----------

def test_weather_settings_need_nothing():
    plugin = next(p for p in catalog.PLUGINS if p["name"] == "get_weather")
    assert plugin["settings"] == "get_weather" and plugin["config"] == "get_weather"
    assert plugin_settings.configured({"plugins": {}}, "get_weather")
    assert plugin_settings.describe({"plugins": {}}, "get_weather")["testable"]


def test_weather_settings_test_button(fake_api):
    r = asyncio.run(plugin_settings.run_test("get_weather", {"location": "天河"}, {"plugins": {}}))
    assert r["place"] == "天河区（广东省广州市）"
    assert r["message"].startswith("查到的地点是 天河区（广东省广州市）。现在（21:00）：阴")

    fake_api.routes["nominatim.openstreetmap.org"] = lambda req: httpx.Response(200, json=[])
    with pytest.raises(ValueError, match="没找到「火星」"):
        asyncio.run(plugin_settings.run_test("get_weather", {"location": "火星"}, {"plugins": {}}))
