import os
from typing import Optional

import requests

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """Get weather information"""
    if not location:
        return "The location parameter cannot be empty"
    if extensions not in ("base", "all"):
        return "Invalid extensions parameter, please enter base or all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "Weather service is not configured (missing AMAP_WEATHER_API or AMAP_API_KEY)"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"Query failed: {data.get('info', 'Unknown error')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"No weather data found for {location}"
            w = lives[0]
            return (
                f"[{w.get('city', location)} Current Weather]\n"
                f"Condition: {w.get('weather', 'Unknown')}\n"
                f"Temperature: {w.get('temperature', 'Unknown')}C\n"
                f"Humidity: {w.get('humidity', 'Unknown')}%\n"
                f"Wind direction: {w.get('winddirection', 'Unknown')}\n"
                f"Wind force: {w.get('windpower', 'Unknown')} level\n"
                f"Updated: {w.get('reporttime', 'Unknown')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"No weather forecast data found for {location}"
        f0 = forecasts[0]
        out = [f"[{f0.get('city', location)} Weather Forecast]", f"Updated: {f0.get('reporttime', 'Unknown')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "Today's weather:",
            f"  Daytime: {today.get('dayweather','Unknown')}",
            f"  Night: {today.get('nightweather','Unknown')}",
            f"  Temperature: {today.get('nighttemp','Unknown')}~{today.get('daytemp','Unknown')}C",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "Error: weather service request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: weather service request failed - {e}"
    except Exception as e:
        return f"Error: failed to parse weather data - {e}"


@tool("get_current_weather")
def get_current_weather_tool(location: str, extensions: Optional[str] = "base") -> str:
    """Get current weather for a city in China."""
    return get_current_weather(location, extensions)
