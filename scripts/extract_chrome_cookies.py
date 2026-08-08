"""One-time helper: pull `auth_token`/`ct0` for x.com out of Chrome's cookie
store and write them into `.env`.

X has retired third-party password login (see README "Authentication"), so
this is the supported way to authenticate: log in to x.com in Chrome once,
then run this script to lift the resulting session cookies into the app's
config, instead of copy-pasting them out of DevTools by hand.

Requires Chrome to not be actively writing to its cookie DB - closing Chrome
first avoids sporadic "database is locked" failures on Windows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import browser_cookie3
except ImportError as exc:
    print("Missing dependency. Run: pip install browser_cookie3", file=sys.stderr)
    raise SystemExit(1) from exc

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
WANTED = ("auth_token", "ct0")


def _find_cookies() -> dict[str, str]:
    """Try both x.com and twitter.com, since older sessions may predate the rename."""
    found: dict[str, str] = {}
    for domain in ("x.com", "twitter.com"):
        try:
            jar = browser_cookie3.chrome(domain_name=domain)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read Chrome cookies for {domain}: {exc}", file=sys.stderr)
            continue
        for cookie in jar:
            if cookie.name in WANTED and cookie.name not in found:
                found[cookie.name] = cookie.value
    return found


def _upsert_env(values: dict[str, str]) -> None:
    """Set X_AUTH_TOKEN / X_CT0 in .env, preserving every other line."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    env_keys = {"auth_token": "X_AUTH_TOKEN", "ct0": "X_CT0"}
    remaining = dict(values)

    for i, line in enumerate(lines):
        match = re.match(r"^(X_AUTH_TOKEN|X_CT0)=", line)
        if not match:
            continue
        key = match.group(1)
        cookie_name = next(k for k, v in env_keys.items() if v == key)
        if cookie_name in remaining:
            lines[i] = f"{key}={remaining.pop(cookie_name)}"

    for cookie_name, value in remaining.items():
        lines.append(f"{env_keys[cookie_name]}={value}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cookies = _find_cookies()
    missing = [name for name in WANTED if name not in cookies]
    if missing:
        print(
            f"Could not find {', '.join(missing)} in Chrome's cookie store for "
            "x.com/twitter.com. Make sure you're logged in to x.com in Chrome, "
            "close Chrome, and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _upsert_env(cookies)
    print(f"Wrote X_AUTH_TOKEN and X_CT0 to {ENV_PATH}")


if __name__ == "__main__":
    main()
