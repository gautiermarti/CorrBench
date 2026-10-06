"""Exercise the local static leaderboard with Playwright; never publishes it."""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def check(args):
    data = json.loads((args.directory / "results.json").read_text())
    errors = []
    args.output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=str(args.chromium), headless=True, args=["--no-sandbox"]
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.url, wait_until="networkidle")
        await page.wait_for_selector("#table-body tr")
        for protocol in data["protocols"]:
            await page.select_option("#protocol", protocol["id"])
            count = sum(r["protocol"] == protocol["id"] for r in data["records"])
            assert await page.locator("#table-body tr").count() == max(count, 1)
            assert (
                await page.locator("#protocol-title").inner_text() == protocol["title"]
            )
            for path in [protocol["protocol_file"], protocol["evidence_file"]]:
                if path:
                    assert (await page.request.get(args.url + path)).status == 200
            for width in [390, 320]:
                await page.set_viewport_size({"width": width, "height": 844})
                assert await page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                ), f"Protocol {protocol['id']} overflows at width {width}"
            await page.set_viewport_size({"width": 1440, "height": 1100})
        await page.select_option("#protocol", "abide_utility_20261005")
        await page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        async with page.expect_download() as info:
            await page.click("#download")
        download = await info.value
        await download.save_as(str(args.output / "downloaded_records.json"))
        downloaded = json.loads((args.output / "downloaded_records.json").read_text())
        assert all(
            row["protocol"] == "abide_utility_20261005" for row in downloaded["records"]
        )
        assert len(downloaded["records"]) == 7
        for width in [390, 320]:
            await page.set_viewport_size({"width": width, "height": 844})
            assert await page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), f"Page overflows at width {width}"
            await page.screenshot(
                path=str(args.output / f"mobile_{width}.png"), full_page=True
            )
        assert not errors, errors
        audit = {
            "browser": browser.version,
            "protocols_checked": len(data["protocols"]),
            "records_checked": len(data["records"]),
            "evidence_links_checked": True,
            "download_verified": True,
            "mobile_widths_without_overflow": [390, 320],
            "page_errors": errors,
        }
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        await browser.close()
    print(json.dumps(audit))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("leaderboard"))
    p.add_argument("--url", default="http://127.0.0.1:8765/")
    p.add_argument("--chromium", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(check(p.parse_args()))


if __name__ == "__main__":
    main()
