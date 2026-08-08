"""Generate illustrative sample JSON/CSV output using the real exporters.

Run manually (`python scripts/generate_sample_output.py`) to refresh the
`data/json/sample_profiles.json` and `data/csv/sample_users.csv` fixtures
checked in for documentation purposes. These are fixture profiles, not
live-scraped data - no network access or credentials are involved.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.exporter import CSVExporter, JSONExporter
from app.models import UserProfile

SAMPLE_PROFILES = [
    UserProfile(
        id="44196397",
        username="elonmusk",
        display_name="Elon Musk",
        bio="Mars, cars, chips, and internet tubes",
        location="Mars",
        website="https://tesla.com",
        profile_image="https://pbs.twimg.com/profile_images/sample/elonmusk.jpg",
        banner_image="https://pbs.twimg.com/profile_banners/sample/elonmusk",
        protected=False,
        verified=True,
        followers=200_000_000,
        following=900,
        tweets=45_000,
        likes=30_000,
        media_count=5_000,
        created_at=datetime(2009, 6, 2, 20, 12, 29, tzinfo=timezone.utc),
        pinned_tweet_id="1234567890123456789",
        language="en",
        is_blue_verified=True,
        profile_url="https://x.com/elonmusk",
    ),
    UserProfile(
        id="2244994945",
        username="openai",
        display_name="OpenAI",
        bio="Creating safe AGI that benefits all of humanity.",
        location="San Francisco, CA",
        website="https://openai.com",
        profile_image="https://pbs.twimg.com/profile_images/sample/openai.jpg",
        banner_image="https://pbs.twimg.com/profile_banners/sample/openai",
        protected=False,
        verified=True,
        followers=8_500_000,
        following=200,
        tweets=3_200,
        likes=1_500,
        media_count=600,
        created_at=datetime(2013, 12, 6, 0, 0, 0, tzinfo=timezone.utc),
        pinned_tweet_id=None,
        language="en",
        is_blue_verified=True,
        profile_url="https://x.com/openai",
    ),
    UserProfile(
        id="18072429",
        username="satyanadella",
        display_name="Satya Nadella",
        bio="Chairman and CEO, Microsoft",
        location="Redmond, WA",
        website="https://microsoft.com",
        profile_image="https://pbs.twimg.com/profile_images/sample/satyanadella.jpg",
        banner_image=None,
        protected=False,
        verified=True,
        followers=3_100_000,
        following=200,
        tweets=6_800,
        likes=900,
        media_count=300,
        created_at=datetime(2009, 1, 8, 0, 0, 0, tzinfo=timezone.utc),
        pinned_tweet_id="9876543210987654321",
        language="en",
        is_blue_verified=True,
        profile_url="https://x.com/satyanadella",
    ),
]


async def main() -> None:
    root = Path(__file__).resolve().parent.parent

    json_exporter = JSONExporter(output_dir=root / "data" / "json")
    csv_exporter = CSVExporter(output_dir=root / "data" / "csv", filename="sample_users.csv")

    json_path = await json_exporter.export(SAMPLE_PROFILES)
    sample_json_path = json_path.rename(json_path.with_name("sample_profiles.json"))
    csv_path = await csv_exporter.export(SAMPLE_PROFILES)

    print(f"Wrote sample JSON to {sample_json_path}")
    print(f"Wrote sample CSV to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
