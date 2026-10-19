"""
SEC 10-K Downloader for BGV Practicum
======================================
Downloads the most recent 10-K filing for Microsoft, NVIDIA, and Alphabet
directly from SEC EDGAR.

Usage:
    1. Edit USER_AGENT below to include your real name and email (SEC requires this)
    2. Run with --probe first to verify which filings will be downloaded:
           python3 download_10k.py --probe
    3. Run without flags to actually download:
           python3 download_10k.py

Output structure:
    sec_10k_data/
        filings_manifest.json     <- summary of all downloaded filings
        microsoft/
            filing_info.json      <- metadata for this filing
            10k.html              <- the raw 10-K HTML document
        nvidia/
            filing_info.json
            10k.html
        alphabet/
            filing_info.json
            10k.html

SEC API Notes:
    - User-Agent MUST contain contact info (name + email), or SEC returns 403
    - Rate limit is 10 requests/second; this script sleeps 0.3s between calls
    - CIKs must be 10-digit zero-padded for the data.sec.gov submissions API
    - Filing documents are served from www.sec.gov/Archives/edgar/data/...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# IMPORTANT: SEC requires a real contact email in the User-Agent.
# Replace this with your actual name and SCU email before running.
USER_AGENT = "Xipeng Ma contact@example.com"

# Target companies and their EDGAR CIKs (10-digit zero-padded)
COMPANIES = {
    "microsoft": {"cik": "0000789019", "display_name": "Microsoft Corporation"},
    "nvidia":    {"cik": "0001045810", "display_name": "NVIDIA Corporation"},
    "alphabet":  {"cik": "0001652044", "display_name": "Alphabet Inc."},
}

OUTPUT_DIR = Path("sec_10k_data")

# Conservative delay to stay well under SEC's 10 req/sec limit
REQUEST_DELAY_SECONDS = 0.3

# -----------------------------------------------------------------------------
# HTTP HELPERS
# -----------------------------------------------------------------------------

def _data_sec_headers():
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

def _www_sec_headers():
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }

def _get(url, headers, expect_json=False):
    time.sleep(REQUEST_DELAY_SECONDS)
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code == 403:
        raise RuntimeError(
            f"SEC returned 403 Forbidden. Most likely cause: User-Agent header "
            f"is missing or doesn't contain real contact info.\n"
            f"  Current User-Agent: {USER_AGENT}\n"
            f"  Edit USER_AGENT at the top of this script to include your name and email."
        )
    r.raise_for_status()
    return r.json() if expect_json else r.content

# -----------------------------------------------------------------------------
# EDGAR API CALLS
# -----------------------------------------------------------------------------

def fetch_company_submissions(cik):
    """Get all recent filings for a company from EDGAR's submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return _get(url, _data_sec_headers(), expect_json=True)

def find_latest_10k(submissions):
    """Locate the most recent 10-K in a submissions response."""
    recent = submissions["filings"]["recent"]
    forms = recent["form"]
    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "form": form,
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            }
    return None

def build_document_url(cik, accession_number, primary_document):
    """Build the URL to the primary 10-K document file."""
    # CIK in the Archives URL is the integer form (no leading zeros)
    cik_int = str(int(cik))
    # Accession number in URL has no dashes
    acc_no_dashes = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_no_dashes}/{primary_document}"
    )

def download_document(url):
    """Download the raw bytes of a filing document."""
    return _get(url, _www_sec_headers(), expect_json=False)

# -----------------------------------------------------------------------------
# MAIN WORKFLOWS
# -----------------------------------------------------------------------------

def probe():
    """Stage 1: show what would be downloaded without actually downloading."""
    print("=" * 70)
    print("PROBE MODE: identifying latest 10-K for each company")
    print("=" * 70)

    if "@" not in USER_AGENT or "example" in USER_AGENT.lower():
        print(f"\nWARNING: USER_AGENT looks like a placeholder: '{USER_AGENT}'")
        print("Edit USER_AGENT at the top of this script before running.\n")

    summary = []
    for slug, info in COMPANIES.items():
        cik = info["cik"]
        print(f"\n>>> {info['display_name']} (CIK {cik})")
        try:
            submissions = fetch_company_submissions(cik)
        except Exception as e:
            print(f"    ERROR: failed to fetch submissions: {e}")
            continue

        latest = find_latest_10k(submissions)
        if latest is None:
            print(f"    No 10-K filings found in recent submissions.")
            continue

        doc_url = build_document_url(cik, latest["accession_number"], latest["primary_document"])
        print(f"    Filed:        {latest['filing_date']}")
        print(f"    Period end:   {latest['report_date']}")
        print(f"    Accession#:   {latest['accession_number']}")
        print(f"    Document:     {latest['primary_document']}")
        print(f"    URL:          {doc_url}")

        summary.append({
            "slug": slug,
            "company": info["display_name"],
            "filing_date": latest["filing_date"],
            "report_date": latest["report_date"],
            "url": doc_url,
        })

    print("\n" + "=" * 70)
    print("Summary: filings that will be downloaded")
    print("=" * 70)
    for s in summary:
        print(f"  {s['company']:30s}  period={s['report_date']}  filed={s['filing_date']}")
    print("\nIf these look right, re-run without --probe to actually download.")
    print("If a fiscal_year doesn't look like 2025, check that company's fiscal year")
    print("end date - NVIDIA and Microsoft have non-calendar fiscal years.")

def download_all():
    """Stage 2: actually download all three 10-Ks and save to disk."""
    print("=" * 70)
    print("DOWNLOAD MODE: fetching latest 10-K for each company")
    print("=" * 70)

    if "@" not in USER_AGENT or "example" in USER_AGENT.lower():
        print(f"\nERROR: USER_AGENT looks like a placeholder: '{USER_AGENT}'")
        print("Edit USER_AGENT at the top of this script before running.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for slug, info in COMPANIES.items():
        cik = info["cik"]
        company_dir = OUTPUT_DIR / slug
        company_dir.mkdir(exist_ok=True)

        print(f"\n>>> {info['display_name']} (CIK {cik})")

        # Step 1: get submissions metadata
        try:
            submissions = fetch_company_submissions(cik)
        except Exception as e:
            print(f"    ERROR fetching submissions: {e}")
            continue

        # Step 2: find latest 10-K
        latest = find_latest_10k(submissions)
        if latest is None:
            print(f"    No 10-K found, skipping.")
            continue
        print(f"    Found 10-K filed {latest['filing_date']} (period {latest['report_date']})")

        # Step 3: download primary document
        doc_url = build_document_url(cik, latest["accession_number"], latest["primary_document"])
        try:
            doc_bytes = download_document(doc_url)
        except Exception as e:
            print(f"    ERROR downloading 10-K document: {e}")
            continue

        # Step 4: save to disk
        # Determine extension from primary_document filename
        ext = os.path.splitext(latest["primary_document"])[1].lower() or ".html"
        doc_path = company_dir / f"10k{ext}"
        doc_path.write_bytes(doc_bytes)

        # Save filing metadata alongside the document
        filing_info = {
            "company": info["display_name"],
            "cik": cik,
            "form": latest["form"],
            "filing_date": latest["filing_date"],
            "report_date": latest["report_date"],
            "accession_number": latest["accession_number"],
            "primary_document": latest["primary_document"],
            "source_url": doc_url,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_size_bytes": len(doc_bytes),
        }
        info_path = company_dir / "filing_info.json"
        info_path.write_text(json.dumps(filing_info, indent=2))

        print(f"    Saved {len(doc_bytes):,} bytes to {doc_path}")
        manifest.append(filing_info)

    # Write top-level manifest
    manifest_path = OUTPUT_DIR / "filings_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{'=' * 70}")
    print(f"Manifest written: {manifest_path}")
    print(f"Downloaded {len(manifest)}/{len(COMPANIES)} filings")
    print("=" * 70)

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings for BGV practicum")
    parser.add_argument("--probe", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    if args.probe:
        probe()
    else:
        download_all()

if __name__ == "__main__":
    main()
