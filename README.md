# squeezer
Squeeze your meta analysis from papers

## Setup & Installation

1. **Install Python Requirements**
   ```bash
   # Install requirements
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API Key**
   - Create a file named `.env` in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

3. **Prepare Your Papers**
   - Create a directory named `papers` in the project root:
     ```bash
     mkdir papers
     ```
   - Place your PDF papers in the `papers` directory
   - Make sure your papers are in PDF format

4. **Run the Program**
   ```bash
   python main.py
   ```
   This will:
   - Process all PDFs in the `papers` directory
   - Extract metrics and metadata from each paper
   - Create an `output.csv` file with the results

## Directory Structure

```
.
├── papers
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
├── output.csv
└── README.md
```

## Acknowledgment
This repository is dedicated to my esteemed political scientist, [@gskulski](https://github.com/gskulski), who tirelessly fights against misinformation to improve political life for humanity.
