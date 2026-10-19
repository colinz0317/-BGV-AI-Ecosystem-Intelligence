import sys
from paths import *

def verify():
    all_files = [
        FILE_ALL_CHUNKS,
        FILE_PORTFOLIO_JSON,
        FILE_PORTFOLIO_JSONL,
        FILE_GAPS_EXTRACTED,
        FILE_GAPS_CRITIC_FILTERED,
        FILE_GAPS_VALID,
        FILE_GMS_RESULTS_JSON,
        FILE_GMS_RESULTS_JSONL,
        FILE_CAPABILITY_LOOKUP,
        FILE_INDUSTRY_LOOKUP,
        FILE_ISS_THEME_MATRIX,
        FILE_ISS_SCORES,
    ]
    missing = []
    for f in all_files:
        if not f.exists():
            missing.append(str(f.relative_to(BASE_DIR)))
    
    if missing:
        print(f"🚨 Missing {len(missing)} files:")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)
    else:
        print("✅ All core files verified and successfully matched with paths.py!")
        sys.exit(0)

if __name__ == "__main__":
    verify()
