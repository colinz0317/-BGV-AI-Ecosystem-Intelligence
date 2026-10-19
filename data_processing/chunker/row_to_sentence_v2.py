"""
BGV Row-to-Sentence Templating (Updated)
=========================================
Converts structured CVC and SEC data rows into natural language chunks
with full metadata for downstream ISS scoring and Pinecone storage.

Usage:
    python row_to_sentence.py

Input:  cvc_cleaned.csv, sec_cleaned.csv
Output: cvc_chunks_output.csv, sec_chunks_output.csv
"""

import pandas as pd
import os

# ============================================================
# CVC DATA → SENTENCE CHUNKS
# ============================================================

def process_cvc_data(file_path, output_path="cvc_chunks_output.csv"):
    print("Processing CVC data...")
    df = pd.read_csv(file_path)
    df = df.fillna("Undisclosed")
    
    def cvc_template(row):
        """
        Template design rationale:
        - Opens with date + company for temporal/entity grounding
        - Includes investment target + amount for specificity
        - Ends with category context for theme classification
        - Readable by both LLM (for labeling) and embedding model (for retrieval)
        """
        date = row['date']
        company = row['company']
        event_type = row['event_type']
        category = row['category']
        target = row['target_company']
        amount = row['amount']
        title = row['title']
        desc = row['description']
        
        # Build the core sentence
        sentence = f"On {date}, {company} engaged in a {event_type} activity"
        
        # Add target company if known
        if target != "Undisclosed" and target != "Unknown":
            sentence += f" involving {target}"
        
        # Add amount if available
        if amount != "Undisclosed":
            sentence += f" with a reported value of {amount}"
        
        sentence += f", within the {category} sector."
        
        # Add title as context
        if title != "Undisclosed" and title:
            sentence += f" Summary: {title}."
        
        # Add description for richer context
        if desc != "Undisclosed" and desc and len(str(desc)) > 30:
            sentence += f" Details: {desc}"
        
        return sentence
    
    chunks = []
    for idx, row in df.iterrows():
        text_chunk = cvc_template(row)
        
        chunks.append({
            # Chunk identification
            "chunk_id": row.get('chunk_id', f"cvc_{idx+1:04d}"),
            
            # The actual text chunk for embedding
            "text_chunk": text_chunk,
            
            # === Metadata for downstream ISS scoring & Pinecone filtering ===
            "company": row['company'],
            "event_date": row['date'],
            "event_type": row['event_type'],           # Maps to ISS dimensions A-E
            "category": row['category'],               # For theme classification
            "signal_strength": row.get('signal_strength', 3),  # Preliminary 1-5
            
            # Additional context (not used in scoring, but useful for traceability)
            "target_company": row.get('target_company', 'Unknown'),
            "amount": row.get('amount', 'Undisclosed'),
            "source_url": row.get('source_url', ''),
            "source_type": row.get('source_type', 'CVC'),
        })
    
    df_out = pd.DataFrame(chunks)
    df_out.to_csv(output_path, index=False)
    print(f"CVC chunking complete: {len(df_out)} chunks → {output_path}\n")
    return df_out


# ============================================================
# SEC FINANCIAL DATA → SENTENCE CHUNKS
# ============================================================

def process_sec_data(file_path, output_path="sec_chunks_output.csv"):
    print("Processing SEC financial data...")
    df = pd.read_csv(file_path)
    df = df.fillna("")
    
    def sec_template(row):
        """
        Template design rationale:
        - Includes company + ticker for entity disambiguation
        - Formats large numbers with commas for readability
        - Adds YoY growth context when available
        - Filing type provides document provenance
        """
        company = row['company_name']
        ticker = row['ticker']
        metric = row['metric_name']
        value = row['value']
        period = row['fiscal_period']
        year = row['fiscal_year']
        form = row['form']
        yoy = row['yoy_growth']
        
        # Format large numbers with commas
        formatted_value = value
        if value != "":
            try:
                formatted_value = f"{float(value):,.0f}"
            except (ValueError, TypeError):
                pass
        
        sentence = (
            f"For {period} of Fiscal Year {year}, {company} ({ticker}) "
            f"filed a {form} report stating its {metric} was {formatted_value}."
        )
        
        # Add YoY growth if available and non-empty
        if yoy != "" and str(yoy).strip() != "":
            try:
                yoy_val = float(yoy)
                direction = "increase" if yoy_val > 0 else "decrease"
                sentence += (
                    f" This represents a {abs(yoy_val):.1f}% year-over-year {direction}."
                )
            except (ValueError, TypeError):
                sentence += f" The Year-over-Year growth was {yoy}%."
        
        return sentence
    
    chunks = []
    for idx, row in df.iterrows():
        text_chunk = sec_template(row)
        
        # Derive event_date from fiscal period + year
        event_date = ""
        if row['fiscal_year'] != "":
            event_date = f"{row['fiscal_year']}-{row['fiscal_period']}" if row['fiscal_period'] else str(row['fiscal_year'])
        
        chunks.append({
            "chunk_id": f"sec_{idx+1:04d}",
            "text_chunk": text_chunk,
            
            # Metadata
            "company": row['company_name'],
            "event_date": event_date,
            "event_type": "Commercial Traction",  # SEC filings map to ISS dimension D
            "category": "Financial Disclosure",
            "signal_strength": 4,                 # SEC filings are high-reliability sources
            
            "source_type": "SEC Filing",
            "form_type": row.get('form', ''),
            "metric_name": row.get('metric_name', ''),
        })
    
    df_out = pd.DataFrame(chunks)
    df_out.to_csv(output_path, index=False)
    print(f"SEC chunking complete: {len(df_out)} chunks → {output_path}\n")
    return df_out


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process CVC data
    cvc_path = os.path.join(current_dir, "cvc_cleaned.csv")
    if os.path.exists(cvc_path):
        process_cvc_data(cvc_path)
    else:
        print(f"Warning: {cvc_path} not found. Run cvc_scraper.py + cvc_cleaner.py first.\n")
    
    # Process SEC data
    sec_path = os.path.join(current_dir, "sec_cleaned.csv")
    if os.path.exists(sec_path):
        process_sec_data(sec_path)
    else:
        print(f"Warning: {sec_path} not found.\n")
    
    print("Row-to-Sentence processing complete.")
