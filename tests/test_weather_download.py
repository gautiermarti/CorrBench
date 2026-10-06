"""Exercise the truncated-transfer failure observed during the NOAA pilot."""

import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_weather", Path(__file__).parents[1] / "research/fetch_ncep_temperature.py"
)
fetch_weather = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_weather)


class Response(io.BytesIO):
    def __init__(self, data, *, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers


@pytest.mark.parametrize("truncated", [False, True])
def test_ranges_require_exact_content_length(tmp_path, monkeypatch, truncated):
    payload = b"\x89HDF\r\n\x1a\n" + b"0" * (3 * 1024 * 1024)

    def fake_open(request, timeout):
        if request.get_method() == "HEAD":
            return Response(
                b"", status=200, headers={"Content-Length": str(len(payload))}
            )
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        data = payload[start : end + 1]
        if truncated:
            data = data[:-1]
        return Response(
            data,
            status=206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    monkeypatch.setattr(fetch_weather.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(fetch_weather.time, "sleep", lambda _seconds: None)
    if truncated:
        with pytest.raises(ValueError, match="Truncated byte range"):
            fetch_weather.fetch(1981, tmp_path, {})
        assert not list(tmp_path.iterdir())
    else:
        result = fetch_weather.fetch(1981, tmp_path, {})
        assert result["bytes"] == result["expected_content_length"] == len(payload)
        assert (tmp_path / result["filename"]).read_bytes() == payload
