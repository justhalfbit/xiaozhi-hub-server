"""查天气（get_weather）：默认用 Open-Meteo（免费、不用注册、不用 Key），配了自己的和风天气 Key 可以继续用和风。

原来只有和风一种，默认用写在代码和 config.yaml 里、全项目共用的 Key；这个 Key 现在一直返回
"Authentication failed, check your KEY/Token or Host"，没自己申请 Key 的人问天气都失败。

数据源（plugins.get_weather.provider）：
- auto（默认）：填了自己的和风 api_key 就用和风，否则用 Open-Meteo。以前配好和风的不用改配置；
- open-meteo：https://open-meteo.com ，地名搜索 + 数值预报，不用 Key；
- qweather：和风天气，要 api_key 和 api_host（和风控制台里每个账号自己的 API Host）。

地点怎么定（两种数据源一样）：
- 配了 plugins.get_weather.local（经纬度，可选）：没给地点，或地点是本地的名字（name、aliases、网关按 IP 查到的城市）时
  直接用这个经纬度，不查地名；
- 没给地点又没配 local：按客户端 IP 查城市，查不到用 default_location；
- 别处：Open-Meteo 的地名搜索用中文名常常找错（「长沙」排第一的是重庆的一个镇，「大理」找到四川），用英文或拼音就准，
  所以工具说明让模型传英文/拼音，再带上省份（重名多：Suzhou 第一个是安徽宿州）或国家代码。模型还是传了中文、
  又找不到像样的城市时，告诉它换拼音再调一次。

函数名和参数（location、lang）不变：core/utils/prompt_manager.py 拼系统提示时直接 import 本函数查本地天气。
"""
import re
from datetime import date
from typing import TYPE_CHECKING

import httpx
from bs4 import BeautifulSoup
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.utils.util import get_ip_info

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

PROVIDERS = ("open-meteo", "qweather")
# 以前写在代码和 config.yaml 里、全项目共用的和风 Key，已经失效：auto 时当成没填
SHARED_QWEATHER_KEY = "a861d0d5e7bf4ee1a83d9a9e4f96d4da"
DEFAULT_QWEATHER_HOST = "mj7p3y7naa.re.qweatherapi.com"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_CACHE_SECONDS = 20 * 60   # 「现在几度」不能用几个小时前的
OPEN_METEO_TIMEOUT = httpx.Timeout(6.0, connect=3.0)   # prompt_manager 拼提示时一共只等 10 秒
# 中文地名只接受这些级别的城市（首都、省会、地级市、县级市），或者人口够多的
CITY_CODES = {"PPLC", "PPLA", "PPLA2", "PPLA3"}
CITY_POPULATION = 100_000

GET_WEATHER_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "获取某个地点的天气（当前和未来7天）。用户问天气、气温、冷不冷、会不会下雨下雪、要不要带伞、穿什么时使用，"
            "不要凭印象回答，也不要先问用户要不要查。"
            "如果上下文里已经给了本地未来7天天气，问本地天气就直接回答、不调用；否则问本地天气时调用此工具、不传 location。"
            "location 用英文或拼音写城市名，例如 Changsha、Suzhou、Tokyo；中国的城市同时给 province（中文省份名），"
            "外国给 country_code。用户说的是省份就用省会；说的是景点、镇或村，就用它所在的城市或县城。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "英文或拼音城市名，例如 Hangzhou、Dali、Tokyo。查本地时不传",
                },
                "province": {
                    "type": "string",
                    "description": "中国城市所在的省份（中文），例如 湖南、云南、江苏；用来区分重名城市",
                },
                "country_code": {
                    "type": "string",
                    "description": "两位国家代码，例如 CN、JP、US；不传默认 CN",
                },
                "lang": {
                    "type": "string",
                    "description": "返回用户使用的语言code，例如zh_CN/zh_HK/en_US/ja_JP等，默认zh_CN",
                },
            },
            "required": ["lang"],
        },
    },
}

# ---------- 和风天气 ----------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    )
}

# 天气代码 https://dev.qweather.com/docs/resource/icons/#weather-icons
WEATHER_CODE_MAP = {
    "100": "晴",
    "101": "多云",
    "102": "少云",
    "103": "晴间多云",
    "104": "阴",
    "150": "晴",
    "151": "多云",
    "152": "少云",
    "153": "晴间多云",
    "300": "阵雨",
    "301": "强阵雨",
    "302": "雷阵雨",
    "303": "强雷阵雨",
    "304": "雷阵雨伴有冰雹",
    "305": "小雨",
    "306": "中雨",
    "307": "大雨",
    "308": "极端降雨",
    "309": "毛毛雨/细雨",
    "310": "暴雨",
    "311": "大暴雨",
    "312": "特大暴雨",
    "313": "冻雨",
    "314": "小到中雨",
    "315": "中到大雨",
    "316": "大到暴雨",
    "317": "暴雨到大暴雨",
    "318": "大暴雨到特大暴雨",
    "350": "阵雨",
    "351": "强阵雨",
    "399": "雨",
    "400": "小雪",
    "401": "中雪",
    "402": "大雪",
    "403": "暴雪",
    "404": "雨夹雪",
    "405": "雨雪天气",
    "406": "阵雨夹雪",
    "407": "阵雪",
    "408": "小到中雪",
    "409": "中到大雪",
    "410": "大到暴雪",
    "456": "阵雨夹雪",
    "457": "阵雪",
    "499": "雪",
    "500": "薄雾",
    "501": "雾",
    "502": "霾",
    "503": "扬沙",
    "504": "浮尘",
    "507": "沙尘暴",
    "508": "强沙尘暴",
    "509": "浓雾",
    "510": "强浓雾",
    "511": "中度霾",
    "512": "重度霾",
    "513": "严重霾",
    "514": "大雾",
    "515": "特强浓雾",
    "900": "热",
    "901": "冷",
    "999": "未知",
}




async def fetch_city_info(location, api_key, api_host, adm=None):
    url = f"https://{api_host}/geo/v2/city/lookup"
    params = {"key": api_key, "location": location, "lang": "zh"}
    if adm:
        params["adm"] = adm
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
        response = await client.get(url, params=params, headers=HEADERS)
    data = response.json()
    if data.get("error") is not None:
        logger.bind(tag=TAG).error(
            f"获取天气失败，原因：{data.get('error', {}).get('detail')}"
        )
        return None
    return data.get("location", [])[0] if data.get("location") else None



async def fetch_weather_page(url):
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
        response = await client.get(url, headers=HEADERS)
    return BeautifulSoup(response.text, "html.parser") if response.status_code == 200 else None


def parse_weather_info(soup):
    city_name = soup.select_one("h1.c-submenu__location").get_text(strip=True)

    current_abstract = soup.select_one(".c-city-weather-current .current-abstract")
    current_abstract = (
        current_abstract.get_text(strip=True) if current_abstract else "未知"
    )

    current_basic = {}
    for item in soup.select(
        ".c-city-weather-current .current-basic .current-basic___item"
    ):
        parts = item.get_text(strip=True, separator=" ").split(" ")
        if len(parts) == 2:
            key, value = parts[1], parts[0]
            current_basic[key] = value

    temps_list = []
    for row in soup.select(".city-forecast-tabs__row")[:7]:  # 取前7天的数据
        date = row.select_one(".date-bg .date").get_text(strip=True)
        weather_code = (
            row.select_one(".date-bg .icon")["src"].split("/")[-1].split(".")[0]
        )
        weather = WEATHER_CODE_MAP.get(weather_code, "未知")
        temps = [span.get_text(strip=True) for span in row.select(".tmp-cont .temp")]
        high_temp, low_temp = (temps[0], temps[-1]) if len(temps) >= 2 else (None, None)
        temps_list.append((date, weather, high_temp, low_temp))

    return city_name, current_abstract, current_basic, temps_list


async def qweather_report(location, cfg, lang="zh_CN", province=None):
    """和风天气：查城市、抓预报页，返回 (报告, 出错时给用户的话)。location 也可以是 "经度,纬度"。"""
    from core.utils.cache.manager import cache_manager, CacheType

    api_key = str(cfg.get("api_key") or SHARED_QWEATHER_KEY).strip()
    api_host = str(cfg.get("api_host") or DEFAULT_QWEATHER_HOST).strip()
    weather_cache_key = f"full_weather_{location}_{lang}"
    cached_weather_report = cache_manager.get(CacheType.WEATHER, weather_cache_key)
    if cached_weather_report:
        return cached_weather_report, None

    city_info = await fetch_city_info(location, api_key, api_host, adm=province)
    if not city_info:
        return f"未找到相关的城市: {location}，请确认地点是否正确", None
    soup = await fetch_weather_page(city_info["fxLink"])
    if not soup:
        return None, "请求失败"
    city_name, current_abstract, current_basic, temps_list = parse_weather_info(soup)

    weather_report = f"您查询的位置是：{city_name}\n\n当前天气: {current_abstract}\n"
    if current_basic:
        weather_report += "详细参数：\n"
        for key, value in current_basic.items():
            if value != "0":  # 过滤无效值
                weather_report += f"  · {key}: {value}\n"
    weather_report += "\n未来7天预报：\n"
    for day, weather, high, low in temps_list:
        weather_report += f"{day}: {weather}，气温 {low}~{high}\n"
    weather_report += "\n（如需某一天的具体天气，请告诉我日期）"
    cache_manager.set(CacheType.WEATHER, weather_cache_key, weather_report)
    return weather_report, None


# ---------- Open-Meteo ----------

# WMO 天气代码 https://open-meteo.com/en/docs （页面底部 WMO Weather interpretation codes）
WMO = {
    0: "晴", 1: "大致晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "较强毛毛雨", 56: "冻毛毛雨", 57: "较强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "较强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
    80: "阵雨", 81: "较强阵雨", 82: "强阵雨", 85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴有冰雹", 99: "强雷阵雨伴有冰雹",
}
WEEKDAYS = "一二三四五六日"
DIRECTIONS = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
# 蒲福风级的上限，km/h：低于第 n 个数就是 n 级
BEAUFORT_KMH = [1, 6, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118]
_SUFFIX = re.compile(r"(特别行政区|自治区|自治州|地区|省|市|区|县)$")


def _norm(name) -> str:
    s = re.sub(r"\s+", "", str(name or "")).lower()
    return _SUFFIX.sub("", s)


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in str(s or ""))


async def _get_json(url: str, params: dict):
    async with httpx.AsyncClient(timeout=OPEN_METEO_TIMEOUT) as client:
        response = await client.get(url, params=params, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def pick_place(results, query: str, province: str = "", country_code: str = ""):
    """从 Open-Meteo 地名搜索结果里挑一个；挑不出像样的城市返回 None。"""
    results = [r for r in results or [] if r.get("latitude") is not None and r.get("longitude") is not None]
    cc = (country_code or "").strip().upper()
    if cc:
        results = [r for r in results if (r.get("country_code") or "").upper() == cc]
    prov = _norm(province)[:2]
    if prov:
        results = [r for r in results if prov in _norm(r.get("admin1"))]
    if _has_cjk(query):
        # 中文名搜索不可靠：只认名字对得上、而且确实是个城市的
        q = _norm(query)
        results = [r for r in results if _norm(r.get("name")) == q and (
            r.get("feature_code") in CITY_CODES or (r.get("population") or 0) >= CITY_POPULATION)]
    # Open-Meteo 自己的排序已经考虑了人口和匹配程度，过滤后取第一个
    return results[0] if results else None


def _place_label(p: dict) -> str:
    name = p.get("name") or ""
    extra = [x for x in (p.get("admin1"), p.get("country")) if x and x != name]
    return f"{name}（{'，'.join(extra)}）" if extra else name


def _wind(speed_kmh, direction_deg) -> str:
    if speed_kmh is None:
        return ""
    level = next((n for n, top in enumerate(BEAUFORT_KMH) if speed_kmh < top), 12)
    if level == 0:
        return "无风"
    d = "" if direction_deg is None else DIRECTIONS[int((direction_deg % 360 + 22.5) // 45) % 8] + "风"
    return f"{d}{level}级"


def _day_label(iso: str, today: date) -> str:
    d = date.fromisoformat(iso)
    rel = {0: "今天", 1: "明天", 2: "后天"}.get((d - today).days, "")
    base = f"{d.month}月{d.day}日 周{WEEKDAYS[d.weekday()]}"
    return f"{rel} {base}" if rel else base


def _num(v, digits=0):
    return "?" if v is None else f"{round(v, digits) + 0.0:g}"   # + 0.0：-0 显示成 0


def format_report(label: str, data: dict) -> str:
    """Open-Meteo 的预报写成和和风版同样结构的报告。"""
    cur = data.get("current") or {}
    lines = [f"您查询的位置是：{label}", ""]
    parts = [WMO.get(cur.get("weather_code"), "未知")]
    if cur.get("temperature_2m") is not None:
        parts.append(f"气温 {_num(cur['temperature_2m'])}°C")
    if cur.get("apparent_temperature") is not None:
        parts.append(f"体感 {_num(cur['apparent_temperature'])}°C")
    if cur.get("relative_humidity_2m") is not None:
        parts.append(f"湿度 {_num(cur['relative_humidity_2m'])}%")
    wind = _wind(cur.get("wind_speed_10m"), cur.get("wind_direction_10m"))
    if wind:
        parts.append(wind)
    if cur.get("precipitation"):
        parts.append(f"正在降水 {_num(cur['precipitation'], 1)} 毫米/小时")
    lines.append("当前天气：" + "，".join(parts))

    daily = data.get("daily") or {}
    days = daily.get("time") or []
    if days:
        try:
            today = date.fromisoformat(str(cur.get("time") or days[0])[:10])
        except ValueError:
            today = date.fromisoformat(days[0])
        lines += ["", "未来7天预报："]
        for i, iso in enumerate(days):
            def col(k):
                v = daily.get(k) or []
                return v[i] if i < len(v) else None
            row = (f"{_day_label(iso, today)}：{WMO.get(col('weather_code'), '未知')}，"
                   f"气温 {_num(col('temperature_2m_min'))}~{_num(col('temperature_2m_max'))}°C")
            if col("precipitation_probability_max") is not None:
                row += f"，降水概率 {_num(col('precipitation_probability_max'))}%"
            lines.append(row)
    lines += ["", "（数据来自 Open-Meteo 的数值预报；如需某一天的具体天气，请告诉我日期）"]
    return "\n".join(lines)


async def find_place(location: str, province: str = "", country_code: str = ""):
    """Open-Meteo 地名搜索。中文名去掉"市""省"之类的后缀再搜（「北京市」搜不到，「北京」能）。"""
    query = _norm(location) if _has_cjk(location) else location.strip()
    data = await _get_json(GEOCODE_URL, {"name": query, "count": 10, "language": "zh", "format": "json"})
    return pick_place((data or {}).get("results"), query, province, country_code)


async def forecast(lat: float, lon: float):
    return await _get_json(FORECAST_URL, {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,"
                   "wind_speed_10m,wind_direction_10m,precipitation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto", "forecast_days": 7, "wind_speed_unit": "kmh",
    })


async def open_meteo_report(place: dict, label: str) -> str:
    from core.utils.cache.manager import cache_manager, CacheType

    key = f"open_meteo_{place['latitude']:.2f}_{place['longitude']:.2f}"
    hit = cache_manager.get(CacheType.WEATHER, key)
    if hit:
        return hit
    report = format_report(label, await forecast(place["latitude"], place["longitude"]))
    cache_manager.set(CacheType.WEATHER, key, report, ttl=OPEN_METEO_CACHE_SECONDS)
    return report


# ---------- 选数据源、定地点 ----------

def provider_of(cfg: dict) -> str:
    """配置里的 provider；auto（或没写、写错）时：填了自己的和风 Key 就用和风，否则 Open-Meteo。"""
    p = str(cfg.get("provider") or "auto").strip().lower().replace("_", "-")
    if p == "openmeteo":
        p = "open-meteo"
    if p in PROVIDERS:
        return p
    key = str(cfg.get("api_key") or "").strip()
    own_key = key and key != SHARED_QWEATHER_KEY and not key.startswith("你的")
    return "qweather" if own_key else "open-meteo"


def local_place(conn, cfg: dict):
    """配置里的本地位置 plugins.get_weather.local：返回 (地点, 算作本地的名字)；没配或经纬度不对返回 (None, set())。"""
    local = cfg.get("local") or {}
    if not isinstance(local, dict):
        return None, set()
    try:
        lat, lon = float(local["latitude"]), float(local["longitude"])
    except (KeyError, TypeError, ValueError):
        return None, set()
    names = {_norm(local.get("name"))} | {_norm(a) for a in local.get("aliases") or []}
    try:
        # 网关按公网 IP 查到的城市（prompt_manager 拿它当本地地点来调本函数）
        from core.utils.cache.manager import cache_manager, CacheType
        ip_city = cache_manager.get(CacheType.LOCATION, getattr(conn, "client_ip", None))
        if ip_city:
            names.add(_norm(ip_city))
    except Exception:
        pass
    names.discard("")
    return {"name": local.get("name") or "本地", "latitude": lat, "longitude": lon}, names


def ip_city(conn) -> str:
    """按客户端 IP 查到的城市（和原来一样先看缓存）；查不到返回空。"""
    from core.utils.cache.manager import cache_manager, CacheType

    client_ip = getattr(conn, "client_ip", None)
    if not client_ip:
        return ""
    info = cache_manager.get(CacheType.IP_INFO, client_ip)
    if not info:
        info = get_ip_info(client_ip, logger)
        if info:
            cache_manager.set(CacheType.IP_INFO, client_ip, info)
    return str((info or {}).get("city") or "").strip()


@register_function("get_weather", GET_WEATHER_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def get_weather(conn: "ConnectionHandler", location: str = None, lang: str = "zh_CN",
                      province: str = None, country_code: str = None):
    cfg = (getattr(conn, "config", None) or {}).get("plugins", {}).get("get_weather", {}) or {}
    provider = provider_of(cfg)
    location = (location or "").strip()
    local, local_names = local_place(conn, cfg)
    use_local = bool(local) and (not location or _norm(location) in local_names)
    if not location and not use_local:
        location = ip_city(conn) or str(cfg.get("default_location") or "").strip()
    if not location and not use_local:
        return ActionResponse(Action.REQLLM, "没有配置本地位置，请告诉我要查哪个城市", None)

    try:
        if provider == "qweather":
            where = f"{local['longitude']:.2f},{local['latitude']:.2f}" if use_local else location
            report, error = await qweather_report(where, cfg, lang, province)
            return ActionResponse(Action.REQLLM, report, error)

        if use_local:
            place, label = local, local["name"]
        else:
            cc = country_code or ("CN" if _has_cjk(location) else "")
            place = await find_place(location, province or "", cc)
            if not place:
                hint = "请用英文或拼音城市名再调用一次，并带上 province（中国）或 country_code（外国）" \
                    if _has_cjk(location) or not (province or country_code) else "请换成它所在的城市或县城再试"
                return ActionResponse(Action.REQLLM, f"没找到「{location}」这个地方。{hint}", None)
            label = _place_label(place)
        return ActionResponse(Action.REQLLM, await open_meteo_report(place, label), None)
    except Exception as e:   # 网络不通、超时、返回的不是 JSON
        logger.bind(tag=TAG).error(f"获取天气失败（{provider}）：{e}")
        return ActionResponse(Action.REQLLM, None, "天气服务暂时连不上，稍后再试")
