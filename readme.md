# Adobe India Hackathon - Connecting the Dots

This repository contains my solution for the Adobe India Hackathon, "Connecting the Dots..." challenge.

## Round 1A: Understand Your Document - Outline Extraction

### Approach

For Round 1A, the mission is to extract a structured outline (Title, H1, H2, H3) from PDF documents. My approach leverages a **heuristic-based method** combined with the robust PDF parsing capabilities of `PyMuPDF` (fitz). This strategy was chosen to adhere strictly to the CPU-only execution, 200MB model size limit (avoiding large ML models), and the tight 10-second execution time constraint.

The core steps are:

1.  **Title Extraction**: The solution first attempts to extract the document title from the PDF's metadata. If unavailable or invalid, it falls back to identifying the largest and most prominent text block on the first page as the title.
2.  **Text Span Analysis**: It iterates through every page and extracts all text spans (pieces of text with consistent formatting) along with their properties: text content, font size, boldness status, page number, and vertical position (Y-coordinate).
3.  **Body Text Size Estimation**: A critical heuristic involves automatically determining the most common font size among longer text spans. This is assumed to be the body text font size, providing a baseline for relative font size comparisons.
4.  **Heading Candidate Identification**: Text spans are identified as potential headings if their font size is significantly larger than the estimated body text size, they are bold, relatively short (not full paragraphs), and do not match common patterns of page numbers or headers/footers.
5.  **Hierarchy Mapping**: The unique font sizes of these heading candidates are then sorted. The largest few distinct font sizes are heuristically mapped to H1, H2, and H3 levels, assuming a general hierarchy based on visual prominence.
6.  **Outline Structuring**: The identified headings are then sorted by page number and their vertical position on the page to ensure a logical reading order. A deduplication step removes any redundant entries.

### Models or Libraries Used

- **PyMuPDF (fitz)**: A powerful and fast Python library for PDF processing. It provides detailed access to text properties (font size, bold status, bounding boxes) which are crucial for the heuristic-based heading detection.

### How to Build and Run Your Solution

The solution is containerized using Docker to ensure a consistent execution environment.

#### Prerequisites:

- Docker installed on your system.

#### Build the Docker Image:

Navigate to the root directory of this project (where `Dockerfile` and `main.py` are located) and run the following command:

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```
