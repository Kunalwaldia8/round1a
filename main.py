import fitz  # PyMuPDF
import json
import os
import re

def extract_pdf_outline(pdf_path):
    """
    Extracts the title and a hierarchical outline (H1, H2, H3) from a PDF.
    Focuses on heuristic rules based on font size, boldness, and position.
    """
    document = fitz.open(pdf_path)
    title = ""
    outline = []
    
    # --- 1. Attempt to extract title ---
    # Prioritize metadata title if available and reasonable
    metadata_title = document.metadata.get("title")
    if metadata_title and metadata_title.strip() and len(metadata_title) > 5 and len(metadata_title) < 200:
        title = metadata_title.strip()
    else:
        # Fallback: Find largest text on the first page
        if document.page_count > 0:
            first_page = document[0]
            text_blocks = first_page.get_text("dict")["blocks"]
            
            max_font_size = 0
            candidate_title = ""
            
            for block in text_blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            current_text = span["text"].strip()
                            current_size = span["size"]
                            
                            # Heuristic for title: largest text, reasonably centered, not too short/long
                            # Adjust bbox[0] and bbox[2] for x0 and x1 coordinates to check for centering
                            
                            # A simple approach for centering could be checking if x0 is roughly > 10% and x1 < 90% of page width.
                            # For simplicity, initially prioritize just font size and length.
                            if current_size > max_font_size:
                                max_font_size = current_size
                                candidate_title = current_text
                            elif current_size == max_font_size and len(current_text) > len(candidate_title) and len(current_text) < 200:
                                candidate_title = current_text
            if candidate_title:
                title = candidate_title
                
    # --- 2. Extract potential headings and their properties ---
    all_text_spans = []
    for page_num in range(document.page_count):
        page = document[page_num]
        text_blocks = page.get_text("dict")["blocks"]
        
        for block in text_blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text: # Only process non-empty text
                            all_text_spans.append({
                                "text": text,
                                "font_size": round(span["size"], 2), # Round to avoid float precision issues
                                "is_bold": "bold" in span["font"].lower() or (span["flags"] & 16) > 0, # Flag 16 is FONT_FLAGS_BOLD
                                "page": page_num + 1,
                                "bbox": span["bbox"],
                                "origin_y": span["bbox"][1] # Top Y-coordinate
                            })

    # --- 3. Determine Heading Levels based on Heuristics ---
    
    # Find common body text font size (often the most frequent non-heading font size)
    # This is a critical step for relative sizing.
    font_sizes = [s["font_size"] for s in all_text_spans if len(s["text"].split()) > 5] # Exclude very short spans
    if not font_sizes:
        body_text_size = 10 # Default if no long text found
    else:
        # Simple mode calculation for body_text_size
        from collections import Counter
        body_text_size = Counter(font_sizes).most_common(1)[0][0]
        
    
    unique_heading_candidate_sizes = sorted(list(set([
        s["font_size"] for s in all_text_spans 
        if s["font_size"] > body_text_size * 1.05 and s["is_bold"] # Candidate is larger than body text and bold
        and len(s["text"]) < 100 # Not too long (e.g., full paragraph)
        and not re.match(r'^\s*(\d+|[IVXLCDM]+\.?)\s*$', s["text"]) # Exclude simple numbers (page numbers)
        and not re.match(r'^\s*Page\s+\d+\s*$', s["text"]) # Exclude "Page X"
    ])), reverse=True)

    # Map the largest few unique font sizes to H1, H2, H3
    heading_level_map = {}
    if len(unique_heading_candidate_sizes) > 0:
        heading_level_map[unique_heading_candidate_sizes[0]] = "H1"
    if len(unique_heading_candidate_sizes) > 1:
        heading_level_map[unique_heading_candidate_sizes[1]] = "H2"
    if len(unique_heading_candidate_sizes) > 2:
        heading_level_map[unique_heading_candidate_sizes[2]] = "H3"
    
    # Final filtering and structuring
    final_outline_candidates = []
    for span in all_text_spans:
        level = heading_level_map.get(span["font_size"])
        if level:
            final_outline_candidates.append({
                "level": level,
                "text": span["text"],
                "page": span["page"],
                "origin_y": span["origin_y"] # Keep for sorting
            })
    
    # Sort the outline by page number and then by vertical position (origin_y)
    final_outline_candidates.sort(key=lambda x: (x["page"], x["origin_y"]))
    
    # Deduplicate and ensure logical flow (e.g., removing sub-headings detected as main headings if not appropriate)
    # A simple deduplication based on consecutive identical entries
    unique_outline = []
    seen = set()
    for item in final_outline_candidates:
        key = (item["level"], item["text"], item["page"])
        if key not in seen:
            unique_outline.append({"level": item["level"], "text": item["text"], "page": item["page"]})
            seen.add(key)
    
    # Optional: Further refinement on hierarchy. For example, if an H3 appears before its likely H2 parent.
    # This can be complex and depends on the diversity of PDFs. Start simple for hackathon.
    
    return {
        "title": title,
        "outline": unique_outline
    }


def process_pdfs_in_directory(input_dir, output_dir):
    """
    Processes all PDF files in the input directory and saves the
    extracted outline as JSON in the output directory.
    """
    print(f"Starting PDF processing. Input directory: {input_dir}, Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)

            print(f"Processing {pdf_path}...")
            try:
                extracted_data = extract_pdf_outline(pdf_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(extracted_data, f, indent=4, ensure_ascii=False)
                print(f"Successfully processed {filename}. Output saved to {output_path}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if processed_count == 0:
        print(f"No PDF files found in {input_dir} or no files processed successfully.")


if __name__ == "__main__":
    # These directories will be mounted by Docker
    INPUT_DIR = "/app/input"
    OUTPUT_DIR = "/app/output"
    
    process_pdfs_in_directory(INPUT_DIR, OUTPUT_DIR)