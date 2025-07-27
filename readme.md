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

### Quick Start

1. **Clone the Repository**

```bash
git clone <repository-url>
cd <repository-name>
```

2. **Prepare Your Data**

- Create an `input` directory and place your PDF files in it
- Create an empty `output` directory for the results

```bash
mkdir -p input output
```

3. **Build and Run with Docker**

```bash
# Build the Docker image
docker build -t pdf-extractor:latest .

# Run the container
docker run --rm \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  pdf-extractor:latest
```

The extracted outlines will be saved as JSON files in the `output` directory.

### Input/Output Format

#### Input

- Place your PDF files in the `input` directory
- Supports any valid PDF document

#### Output

JSON files with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Sub Heading",
      "page": 2
    }
  ]
}
```

### Docker Requirements

- Docker Engine 20.10.0 or later
- At least 2GB of available memory
- Internet connection for the initial build (to download base images and dependencies)

### Troubleshooting

1. **Permission Issues**

   ```bash
   # If you encounter permission issues, try running Docker with sudo
   sudo docker build -t pdf-extractor:latest .
   sudo docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" pdf-extractor:latest
   ```

2. **Memory Issues**

   - If the container crashes, ensure Docker has enough memory allocated
   - For Docker Desktop users, increase memory in Docker Desktop settings

3. **No Output Generated**
   - Verify that your PDF files are in the correct `input` directory
   - Check that the files have `.pdf` extension
   - Ensure the files are readable

### Development

If you want to modify the code or run it without Docker:

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Run Directly**

```bash
python main.py
```

### License

This project is licensed under the terms specified in the LICENSE file.
