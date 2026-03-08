"""
Seed script: submits backdated test articles to /analyze-article to generate
historical signals for portfolio backtesting.

Articles are spread from 2025-04 to 2026-02 and cover three macro themes:
  - geopolitical oil supply shocks  (fires geo_oil_supply_shock + inflation_safe_haven + shipping_disruption)
  - shipping lane disruptions        (fires shipping_disruption + geo_oil_supply_shock)
  - OPEC supply management          (fires geo_oil_supply_shock + inflation_safe_haven)

Each article is distinct enough in wording to form its own cluster.

Usage:
    # Make sure the FastAPI server is running first:
    py research/seed_historical_articles.py

    # Dry-run (print payloads, don't POST):
    py research/seed_historical_articles.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

_API_URL = "http://localhost:8000/analyze-article"

# ---------------------------------------------------------------------------
# Article corpus
# Each dict maps to ArticleInput fields. published_at drives the signal date.
# Varied wording ensures low TF-IDF similarity → distinct clusters.
# ---------------------------------------------------------------------------
ARTICLES: list[dict[str, str]] = [
    # === April 2025 — Iran / Hormuz ===
    {
        "title": "Iran Threatens to Close Strait of Hormuz After New US Sanctions",
        "source": "Reuters",
        "published_at": "2025-04-14T08:00:00",
        "body": (
            "Iranian officials warned Monday they may block tanker traffic through the Strait of Hormuz "
            "if the United States proceeds with expanded oil sanctions targeting the Islamic Republic. "
            "The Hormuz strait carries roughly 20 percent of global crude shipments daily. "
            "Brent crude jumped 3.2 percent on the news as traders priced in a potential supply disruption. "
            "OPEC members held emergency consultations to assess spare capacity."
        ),
    },
    {
        "title": "IRGC Seizes Norwegian Oil Tanker Near Hormuz Strait",
        "source": "Bloomberg",
        "published_at": "2025-04-22T10:30:00",
        "body": (
            "Iran's Revolutionary Guard Corps boarded and seized a Norwegian-flagged crude tanker "
            "in the Strait of Hormuz, escalating maritime tensions in the Persian Gulf. "
            "The vessel was carrying 2 million barrels of Iraqi crude bound for Rotterdam. "
            "The seizure marks the third tanker incident in the region this year, pushing WTI crude "
            "to its highest level since January. Energy stocks rallied sharply."
        ),
    },
    # === May 2025 — Houthi / Red Sea ===
    {
        "title": "Houthi Missiles Strike Container Ship in Red Sea, Crew Evacuated",
        "source": "AP",
        "published_at": "2025-05-07T06:00:00",
        "body": (
            "Houthi rebels launched missile strikes against a container vessel in the southern Red Sea, "
            "forcing the crew to abandon ship. The attack is the most severe since the conflict began "
            "and prompted immediate rerouting announcements from major shipping operators. "
            "Freight rates on Asia-to-Europe lanes surged 18 percent within hours. "
            "The incident is expected to extend delivery timelines by two to three weeks."
        ),
    },
    {
        "title": "Shipping Giants Divert Fleets Around Cape of Good Hope Amid Red Sea Danger",
        "source": "FT",
        "published_at": "2025-05-15T09:00:00",
        "body": (
            "The world's largest container shipping operators announced they are permanently rerouting "
            "cargo around the Cape of Good Hope, bypassing the Suez Canal entirely, following a week "
            "of intensified Houthi attacks. The decision adds roughly 10 days to Asia-Europe transit times "
            "and threatens to unravel just-in-time inventory strategies across global retail supply chains. "
            "Dry bulk rates also climbed as tanker operators weighed similar diversions."
        ),
    },
    # === June 2025 — Libya ===
    {
        "title": "Libya's Largest Oil Field Sharara Halts Output After Armed Protest",
        "source": "Reuters",
        "published_at": "2025-06-18T11:00:00",
        "body": (
            "Armed groups blockaded Libya's Sharara oil field, forcing a shutdown of 300,000 barrels "
            "per day of crude production. The facility, operated in partnership with a European major, "
            "had only recently resumed full operations after a previous closure. "
            "The disruption tightened the Atlantic crude market and lifted Brent by 2.1 percent. "
            "OPEC noted the outage as a force-majeure event outside its supply management framework."
        ),
    },
    # === July 2025 — Russia sanctions ===
    {
        "title": "G7 Tightens Russia Crude Price Cap Enforcement, Targets Shadow Fleet",
        "source": "WSJ",
        "published_at": "2025-07-10T07:30:00",
        "body": (
            "G7 finance ministers agreed on enhanced enforcement of the Russian crude oil price cap, "
            "targeting the growing shadow tanker fleet that has helped Moscow circumvent sanctions. "
            "The new measures include port-access bans for vessels suspected of transporting Russian crude "
            "above the $60 cap and expanded secondary sanctions on non-compliant insurers. "
            "Analysts estimate the move could remove up to 800,000 barrels per day from global markets "
            "by year-end, supporting Brent crude prices."
        ),
    },
    # === August 2025 — Venezuela ===
    {
        "title": "US Reinstates Venezuela Oil Sanctions After Disputed Election",
        "source": "Bloomberg",
        "published_at": "2025-08-05T14:00:00",
        "body": (
            "The US Treasury Department reinstated oil sanctions on Venezuela after the Maduro government "
            "refused to certify opposition-verified election results. The renewed sanctions are expected "
            "to cut Venezuelan crude exports by 400,000 barrels per day. "
            "Several Asian refiners that had signed long-term supply agreements scrambled to secure "
            "alternative cargoes. WTI crude rose sharply on concerns about reduced Western Hemisphere supply."
        ),
    },
    # === September 2025 — OPEC+ cut ===
    {
        "title": "OPEC+ Shocks Markets with Surprise 600,000 bpd Production Cut",
        "source": "Reuters",
        "published_at": "2025-09-12T12:00:00",
        "body": (
            "OPEC+ ministers agreed in a surprise online meeting to deepen collective output cuts by "
            "600,000 barrels per day starting in October, citing weakening global demand and rising "
            "non-OPEC supply. Saudi Arabia volunteered a unilateral additional reduction of 250,000 bpd. "
            "Crude oil prices surged five percent in the hour after the announcement. "
            "Energy stocks and XLE rallied strongly while airlines fell on higher fuel cost forecasts."
        ),
    },
    # === October 2025 — Iraq / Turkey pipeline ===
    {
        "title": "Explosion Halts Iraq-Turkey Pipeline, Cutting Kurdish Crude Exports",
        "source": "AP",
        "published_at": "2025-10-20T08:00:00",
        "body": (
            "An explosion near the Kirkuk-Ceyhan pipeline halted approximately 350,000 barrels per day "
            "of Iraqi Kurdish crude exports. Repair crews estimated a two-week restoration timeline. "
            "The disruption hit the Mediterranean crude market hardest, with Azeri Light and Iraqi Basra "
            "Light differentials widening. Energy sector ETFs outperformed as traders anticipated "
            "tighter supply from the Mediterranean basin."
        ),
    },
    # === November 2025 — Middle East escalation ===
    {
        "title": "Middle East Escalation Triggers Surge in Oil Risk Premium",
        "source": "FT",
        "published_at": "2025-11-04T10:00:00",
        "body": (
            "A sudden escalation in regional hostilities across the Middle East sent crude oil prices "
            "sharply higher as traders priced in disruption risk to key shipping corridors. "
            "Tanker insurance rates spiked and several operators suspended sailings through the "
            "region. The geopolitical risk premium on Brent crude reached its highest level in 18 months. "
            "Gold also rallied as investors sought safe-haven assets amid the heightened uncertainty."
        ),
    },
    # === December 2025 — Iran nuclear ===
    {
        "title": "Iran Nuclear Talks Collapse, Broad Sanctions to Resume in 30 Days",
        "source": "WSJ",
        "published_at": "2025-12-08T15:00:00",
        "body": (
            "Negotiations over Iran's nuclear programme collapsed in Vienna, with Western powers "
            "announcing a phased reimposition of oil sanctions within 30 days. "
            "Iran's foreign minister warned of 'consequences' for regional shipping. "
            "The prospect of removing up to 1.5 million barrels per day of Iranian crude from global "
            "markets sent Brent crude surging. Analysts warned the sanctions could trigger retaliatory "
            "action in the Strait of Hormuz."
        ),
    },
    # === January 2026 — Panama Canal ===
    {
        "title": "Panama Canal Restricts Tanker Drafts as Drought Cuts Water Levels",
        "source": "Bloomberg",
        "published_at": "2026-01-06T09:00:00",
        "body": (
            "The Panama Canal Authority imposed new draft restrictions on LNG tankers and large crude "
            "carriers after reservoir water levels dropped to a 30-year low. "
            "Vessels are being forced to reduce cargo loads by up to 40 percent or reroute around Cape Horn. "
            "The disruption is adding pressure to already elevated shipping rates on Pacific routes "
            "and threatens US Gulf Coast crude export schedules. Freight operators warned of further "
            "restrictions if rainfall remains below seasonal averages."
        ),
    },
    # === January 2026 — Houthi escalation 2 ===
    {
        "title": "Houthis Sink Oil Tanker in Gulf of Aden in Major Escalation",
        "source": "Reuters",
        "published_at": "2026-01-22T07:00:00",
        "body": (
            "Houthi rebels sank a Greek-owned crude tanker in the Gulf of Aden using naval mines, "
            "marking the first confirmed sinking of a commercial vessel since the conflict began. "
            "The attack killed three crew members and spilled an estimated 500,000 barrels of crude. "
            "The incident triggered immediate diplomatic condemnation and pushed oil prices to a five-month "
            "high. Multiple energy and shipping companies suspended operations in the region."
        ),
    },
    # === February 2026 — Russia oil ===
    {
        "title": "Fire at Siberian Oil Facility Disrupts Russian Crude Export Pipeline",
        "source": "Bloomberg",
        "published_at": "2026-02-09T11:00:00",
        "body": (
            "A fire at a major oil processing facility in western Siberia forced the temporary shutdown "
            "of a key pipeline feeding Russian crude exports to Europe. Approximately 200,000 barrels "
            "per day of export capacity is expected to be offline for at least two weeks. "
            "The disruption compounded existing concerns about Russian supply after enhanced G7 "
            "price-cap enforcement and supported Brent crude prices in thin early Asian trade."
        ),
    },
    # === February 2026 — Iran OPEC confrontation ===
    {
        "title": "Iran Threatens OPEC+ Exit Over Quota Dispute, Oil Markets on Edge",
        "source": "Reuters",
        "published_at": "2026-02-19T13:00:00",
        "body": (
            "Iran's oil minister stated the country may leave the OPEC+ production framework entirely "
            "if other members do not recognise its exemption from agreed supply cuts. "
            "The standoff threatens to unravel the alliance's output management strategy ahead of the "
            "next scheduled meeting. Crude markets swung sharply as traders assessed the implications "
            "of uncapped Iranian supply flooding the market versus further OPEC+ fragmentation risk."
        ),
    },
    # === Additional articles for better backtest coverage ===
    # === May 2025 — Saudi pipeline attack ===
    {
        "title": "Drone Attack on Saudi Oil Infrastructure Raises Supply Concerns",
        "source": "AP",
        "published_at": "2025-05-28T08:00:00",
        "body": (
            "Drones struck a Saudi Aramco pipeline facility in the Eastern Province, temporarily "
            "interrupting crude flow to the Ras Tanura export terminal. The attack was claimed by "
            "a Yemeni militant group. Though the disruption lasted only hours, it highlighted the "
            "vulnerability of Gulf energy infrastructure. WTI crude jumped over two percent "
            "as traders reassessed geopolitical risk in the Arabian Peninsula."
        ),
    },
    # === August 2025 — Strait of Hormuz naval incident ===
    {
        "title": "US-Iran Naval Standoff in Hormuz Strait Stalls Tanker Traffic",
        "source": "FT",
        "published_at": "2025-08-20T06:30:00",
        "body": (
            "A confrontation between US Navy vessels and Iranian patrol boats in the Strait of Hormuz "
            "caused a temporary halt in commercial tanker traffic as operators awaited clearance. "
            "The standoff lasted six hours but sent insurance war-risk premiums on Persian Gulf routes "
            "to their highest level in three years. Energy traders pushed crude futures sharply higher "
            "on fears of a broader maritime incident affecting the world's most important oil chokepoint."
        ),
    },
    # === October 2025 — Suez Canal shipping backup ===
    {
        "title": "Suez Canal Congestion Mounts as Red Sea Diversions Overwhelm Cape Route",
        "source": "WSJ",
        "published_at": "2025-10-06T10:00:00",
        "body": (
            "A surge in Cape of Good Hope rerouting has created unprecedented congestion at Suez, "
            "with hundreds of container ships forming a queue stretching into the Mediterranean. "
            "The congestion threatens to disrupt European retail supply chains ahead of the holiday "
            "season. Container shipping rates on Asia-Europe lanes climbed to 18-month highs while "
            "dry bulk indices also strengthened as demand for alternative vessel types increased."
        ),
    },
    # === November 2025 — OPEC+ compliance ===
    {
        "title": "OPEC+ Compliance Audit Reveals Iraq, Kazakhstan Overproduction",
        "source": "Reuters",
        "published_at": "2025-11-25T09:00:00",
        "body": (
            "An internal OPEC+ compliance review found Iraq and Kazakhstan exceeded their production "
            "quotas by a combined 450,000 barrels per day in the latest reporting period. "
            "Saudi Arabia and the UAE demanded compensatory cuts, threatening alliance stability. "
            "The dispute, coming ahead of the December ministerial meeting, introduced fresh uncertainty "
            "into crude oil markets. Analysts warned of potential supply increases if the deal unravels."
        ),
    },
    # === January 2026 — Winter demand surge ===
    {
        "title": "Arctic Cold Snap Drains Heating Oil Reserves, Crude Demand Spikes",
        "source": "Bloomberg",
        "published_at": "2026-01-14T12:00:00",
        "body": (
            "An extended Arctic blast across North America and Europe triggered an unexpected surge in "
            "heating oil and natural gas demand, draining winter stockpiles faster than forecast. "
            "US crude oil inventories fell by 9 million barrels in a single week. "
            "The demand shock, combined with limited OPEC spare capacity, pushed WTI crude to its "
            "highest price in four months and lifted energy sector ETFs broadly."
        ),
    },
]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post_article(payload: dict[str, Any], timeout: int = 10) -> dict | None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        _API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"    HTTP {e.code}: {body[:120]}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed historical articles for backtesting.")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads without POSTing.")
    args = parser.parse_args()

    print(f"\nSeeding {len(ARTICLES)} historical articles → {_API_URL}")
    print("─" * 72)

    ok = failed = skipped = 0

    for i, article in enumerate(ARTICLES, 1):
        date_str = article["published_at"][:10]
        print(f"  [{i:>2}/{len(ARTICLES)}] {date_str}  {article['title'][:55]}", end="")

        if args.dry_run:
            print("  [DRY-RUN]")
            skipped += 1
            continue

        result = _post_article(article)
        if result is not None:
            ideas = len(result.get("ideas", []))
            rules = result.get("matched_rule_ids", [])
            print(f"  → {ideas} ideas  rules={rules}")
            ok += 1
        else:
            print("  → FAILED")
            failed += 1

        time.sleep(0.15)  # avoid hammering SQLite

    print("─" * 72)
    print(f"  Submitted: {ok}   Failed: {failed}   Dry-run: {skipped}")

    if ok > 0:
        print("\nNext steps:")
        print("  1. py research/cluster_backfill.py --reset")
        print("  2. py research/analyze_clusters.py --force")
        print("  3. py research/build_market_context.py")
        print("  4. py research/build_signal_rankings.py")
        print("  5. py research/replay_backtest.py")


if __name__ == "__main__":
    main()
