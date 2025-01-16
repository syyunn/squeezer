import os
from PyPDF2 import PdfReader
from openai import OpenAI

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

import tiktoken
from tqdm import tqdm

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Counts the number of tokens in a given text string using the tiktoken library.
    Args:
        text (str): The text for which you want to count tokens.
        model (str): The model name to get the appropriate encoding. 
                     Defaults to "gpt-3.5-turbo".
    Returns:
        int: The count of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # If the exact model is unknown to tiktoken, fallback to a similar encoding
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)

load_dotenv()

class QuoteSchema(BaseModel):
    quote: Optional[str] = None # direct quote from the paper that indicates the Metric
    page_number: Optional[int] = None
    metric_value: Optional[float] = None # the value of the metric, e.g. 0.95, 0.85, 0.75, etc.

class MetricSchema(BaseModel):
    task: Optional[str] = None # description of the predictive task, e.g. "predicting the gender of the author that wrote the reply in the reddit thread"
    accuracy: Optional[QuoteSchema] = None
    recall: Optional[QuoteSchema] = None
    precision: Optional[QuoteSchema] = None
    f1_score: Optional[QuoteSchema] = None # if you can find f1 score, you should also include accuracy, precision, recall because you can reveresely calculate f1 score from accuracy, precision, recall or vice versa
    roc_auc: Optional[QuoteSchema] = None    

class OutputSchema(BaseModel):
    file_name: Optional[str] = None
    paper_title: Optional[str] = None
    paper_authors: Optional[List[str]] = None
    paper_date: Optional[str] = None
    paper_url: Optional[str] = None
    metric: Optional[List[MetricSchema]] # list of metrics for the paper - each metric is for a different task, and a paper can have multiple tasks, so the list can have multiple metrics

output_schema_str = """
class QuoteSchema(BaseModel):
    quote: Optional[str] = None # direct quote from the paper that indicates the Metric
    page_number: Optional[int] = None
    metric_value: Optional[float] = None # the value of the metric, e.g. 0.95, 0.85, 0.75, etc.

class MetricSchema(BaseModel):
    task: Optional[str] = None # description of the predictive task, e.g. "predicting the gender of the author that wrote the reply in the reddit thread"
    accuracy: Optional[QuoteSchema] = None
    recall: Optional[QuoteSchema] = None
    precision: Optional[QuoteSchema] = None
    f1_score: Optional[QuoteSchema] = None # if you can find f1 score, you should also include accuracy, precision, recall because you can reveresely calculate f1 score from accuracy, precision, recall or vice versa
    roc_auc: Optional[QuoteSchema] = None    

class OutputSchema(BaseModel):
    file_name: Optional[str] = None
    paper_title: Optional[str] = None
    paper_authors: Optional[List[str]] = None
    paper_date: Optional[str] = None
    paper_url: Optional[str] = None
    metric: Optional[List[MetricSchema]] # list of metrics for the paper - each metric is for a different task, and a paper can have multiple tasks, so the list can have multiple metrics

"""

def extract_text_from_pdfs(directory='papers'):
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found")
        return
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return
    
    # Process each PDF file with progress bar
    for pdf_file in tqdm(pdf_files, desc="Extracting text from PDFs"):
        file_path = os.path.join(directory, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        
        try:
            # Create PDF reader object
            reader = PdfReader(file_path)
            
            # Extract text from each page
            text_by_page = {}
            for i, page in enumerate(reader.pages, 1):
                text_by_page[f"page{i}"] = page.extract_text()
            
            # Convert the dictionary to string representation
            text = str(text_by_page)
            token_count = count_tokens(text)
            
            # If token count exceeds limit, iteratively remove pages from the back
            while token_count > 120000 and len(text_by_page) > 1:
                # Remove the last page
                last_page = max(text_by_page.keys())
                del text_by_page[last_page]
                # Recalculate token count
                text = str(text_by_page)
                token_count = count_tokens(text)

            print(f"Token count: {token_count}")
            filename = os.path.basename(file_path)
            
            # You can process the text here as needed
            # For now, we'll just print the first 200 characters
            print(f"Extracted text preview: {text[:200]}...")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


system_prompt: str = f"""You are an expert academic researcher tasked with extracting structured information from academic papers.

Your task is to:
1. Carefully read through the provided paper text
2. Extract key information including:
   - Paper title, authors, date, and URL (if available)
   - All quantitative metrics and results
   - For each metric, identify:
     * The specific task being evaluated
     * The exact quote containing the metric (must be copied verbatim from the text)
     * The page number where the metric appears
     * The numerical value of the metric

Guidelines:
- Only extract information that is explicitly stated in the text
- Quotes must be exact, word-for-word copies from the original text, not paraphrased
- Include the complete sentence or passage containing the metric
- If information is missing or unclear, leave those fields as null
- Be precise with numerical values and page numbers
- Separate different tasks/experiments into distinct metric entries

Format all information according to the provided schema structure.

{output_schema_str}
"""

def make_task_prompt(paper_text: str, filename: str) -> str:
    return f"""Please analyze the following academic paper text and extract the required information according to the schema.

Filename: {filename}

Paper Text:
---
{paper_text}
---

Important Notes:
1. Focus on identifying and extracting:
   - Metrics (accuracy, precision, recall, F1, ROC-AUC)
   - The specific tasks these metrics are associated with
   - Direct quotes containing the metrics
   - Page numbers for each metric

2. For each metric found:
   - Ensure the task description clearly explains what is being measured
   - Include the complete quote containing the metric
   - Extract the exact numerical value
   - Note the page number where it appears

3. If you find multiple tasks or experiments, create separate metric entries for each one.

Please provide the extracted information in a structured format matching the schema."""

def process_paper(file_path: str) -> OutputSchema:
    print(f"\nProcessing file: {file_path}")
    try:
        # Create PDF reader object
        reader = PdfReader(file_path)
        
        # Extract text from each page with page numbers
        text_content = []
        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            text_content.append(f"[Page {i}]\n{page_text}")
        
        # Join all pages with clear page demarcation
        full_text = "\n\n".join(text_content)
        
        # Get filename from file_path
        filename = os.path.basename(file_path)
        
        # Create the task prompt with filename and properly formatted text
        task_prompt = make_task_prompt(full_text, filename)
        
        # Create messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is the task prompt:\n"
                            f"{task_prompt}\n\n"
                            "Please convert it to valid JSON strictly matching the schema provided."
                        )
                    }
                ]
            }
        ]
        
        # Make API call and parse response
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",  # or your preferred model
            messages=messages,
            response_format=OutputSchema

        )
        
        return completion.choices[0].message.parsed
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_papers(directory='papers') -> List[OutputSchema]:
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found")
        return []
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return []
    
    results = []
    # Process each PDF file with progress bar
    for pdf_file in tqdm(pdf_files, desc="Processing papers"):
        file_path = os.path.join(directory, pdf_file)
        
        result = process_paper(file_path)
        if result:
            # Print results immediately after processing each paper
            print(f"\n{'='*80}")
            print(f"File: {result.file_name}")
            print(f"Title: {result.paper_title}")
            print(f"Authors: {', '.join(result.paper_authors or ['No authors listed'])}")
            print(f"Date: {result.paper_date or 'No date available'}")
            print(f"URL: {result.paper_url or 'No URL available'}")
            
            if result.metric:
                for i, metric in enumerate(result.metric, 1):
                    print(f"\nTask {i}: {metric.task}")
                    
                    metrics_data = {
                        'Accuracy': metric.accuracy,
                        'Precision': metric.precision,
                        'Recall': metric.recall,
                        'F1 Score': metric.f1_score,
                        'ROC AUC': metric.roc_auc
                    }
                    
                    for metric_name, metric_data in metrics_data.items():
                        if metric_data:
                            print(f"\n{metric_name}:")
                            print(f"  Value: {metric_data.metric_value}")
                            print(f"  Page: {metric_data.page_number}")
                            print(f"  Quote: \"{metric_data.quote}\"")
            else:
                print("\nNo metrics found in this paper")
            
            print(f"{'='*80}")
            
            results.append(result)
    
    return results

def create_csv_output(results: List[OutputSchema], output_file: str = 'output.csv'):
    import pandas as pd
    
    rows = []
    for paper in results:
        # Get paper metadata
        paper_meta = {
            'file_name': paper.file_name,
            'paper_title': paper.paper_title,
            'paper_authors': '; '.join(paper.paper_authors) if paper.paper_authors else None,
            'paper_date': paper.paper_date,
            'paper_url': paper.paper_url
        }
        
        print("Processing paper: ", paper_meta)
        
        # If no metrics, create one row with just metadata
        if not paper.metric:
            rows.append(paper_meta)
        else:
            # For each task in the paper, create a row
            for metric in paper.metric:
                row = paper_meta.copy()
                row['task'] = metric.task
                
                # Add metric values and their associated quotes and page numbers
                for metric_name in ['accuracy', 'recall', 'precision', 'f1_score', 'roc_auc']:
                    metric_data = getattr(metric, metric_name)
                    if metric_data:
                        row[f'{metric_name}_value'] = metric_data.metric_value
                        row[f'{metric_name}_quote'] = metric_data.quote
                        row[f'{metric_name}_page'] = metric_data.page_number
                    else:
                        row[f'{metric_name}_value'] = None
                        row[f'{metric_name}_quote'] = None
                        row[f'{metric_name}_page'] = None
                
                rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    results = process_all_papers()
    create_csv_output(results)
        