import os
import streamlit as st
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup, MultifieldParser
from whoosh import highlight
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import string

# Setup directories
UPLOAD_DIR = "uploads"
INDEX_DIR = "indexdir"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Define schema
schema = Schema(filename=ID(stored=True), content=TEXT(stored=True))

# Initialize index
if not os.listdir(INDEX_DIR):
    ix = create_in(INDEX_DIR, schema)
else:
    ix = open_dir(INDEX_DIR)

# Language processing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Add custom CSS for highlighting
st.markdown("""
    <style>
    .highlight {
        color: #ff0000;
        font-weight: bold;
        background-color: #ffebee;
        padding: 2px 4px;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    pos_tags = pos_tag(filtered_tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return ' '.join(lemmatized).strip()


# Extraction functions
def extract_pdf_pages(file_path):
    doc = fitz.open(file_path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def extract_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            st.write(f"Successfully read TXT file: {file_path}")
            return content
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin1') as f:
                content = f.read()
                st.write(f"Read TXT file with latin1 encoding: {file_path}")
                return content
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return ""


def extract_csv_rows(file_path):
    df = pd.read_csv(file_path)
    return [(i + 1, ' '.join([str(cell) for cell in row if pd.notna(cell)])) for i, row in df.iterrows()]


def extract_excel_rows(file_path):
    xls = pd.ExcelFile(file_path)
    rows = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        for i, row in df.iterrows():
            content = ' '.join([str(cell) for cell in row if pd.notna(cell)])
            rows.append((sheet_name, i + 1, content))
    return rows


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f'{name}{a}_')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, f'{name}{i}_')
        else:
            out[name[:-1]] = str(x)

    flatten(y)
    return ' '.join(out.values())


def extract_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [flatten_json(item) for item in data]
    else:
        return [flatten_json(data)]


def extract_web_content(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        title = soup.title.string if soup.title else ""
        paragraphs = ' '.join(p.get_text() for p in soup.find_all('p'))
        return f"{title} {paragraphs}"
    except Exception:
        return ""


# Streamlit Interface
st.title("🔍 Multi-format Search Engine")

# Add query help
with st.expander("ℹ️ Search Query Help"):
    st.markdown("""
    **Advanced Search Syntax:**
    - **AND**: Use space between words (e.g., `python AND data`)
    - **OR**: Use `OR` between words (e.g., `python OR java`)
    - **Phrase**: Use quotes (e.g., `"machine learning"`)
    - **Wildcard**: Use `*` (e.g., `comput*` for computer, computing, etc.)
    - **Fuzzy**: Use `~` (e.g., `roam~` for roam, foam, etc.)
    - **Grouping**: Use parentheses (e.g., `(python OR java) AND data`)
    """)

file_types = ["PDF", "TXT", "CSV", "Excel", "JSON", "Web"]
selected_types = st.multiselect("Select file type(s) for search", file_types, default=file_types)

uploaded_files = st.file_uploader("Upload files (PDF, TXT, CSV, Excel, JSON)",
                                  type=["pdf", "txt", "csv", "xlsx", "json"], accept_multiple_files=True)
urls_input = st.text_area("Enter web URLs, one per line.")
urls = [u.strip() for u in urls_input.split("\n") if u.strip()]

if st.button("⚙️ Index Files"):
    with ix.writer() as writer:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            name = uploaded_file.name.lower()
            ext = os.path.splitext(name)[1]

            if ext == ".pdf":
                for page_num, text in extract_pdf_pages(file_path):
                    writer.add_document(
                        filename=f"{uploaded_file.name} (Page {page_num})",
                        content=preprocess(text)
                    )
            elif ext == ".txt":
                text = extract_txt(file_path)
                st.write(f"Processing TXT file: {uploaded_file.name}")
                st.write(f"Content length: {len(text)} characters")
                if text.strip():  # Only add if there's actual content
                    writer.add_document(
                        filename=uploaded_file.name,
                        content=preprocess(text)
                    )
                    st.write(f"Successfully indexed TXT file: {uploaded_file.name}")
                else:
                    st.warning(f"TXT file is empty: {uploaded_file.name}")
            elif ext == ".csv":
                for row_num, row_text in extract_csv_rows(file_path):
                    writer.add_document(
                        filename=f"{uploaded_file.name} (Row {row_num})",
                        content=preprocess(row_text)
                    )
            elif ext == ".xlsx":
                for sheet, row_num, row_text in extract_excel_rows(file_path):
                    writer.add_document(
                        filename=f"{uploaded_file.name} ({sheet} - Row {row_num})",
                        content=preprocess(row_text)
                    )
            elif ext == ".json":
                for i, entry in enumerate(extract_json(file_path)):
                    writer.add_document(
                        filename=f"{uploaded_file.name} (Entry {i + 1})",
                        content=preprocess(entry)
                    )

        for url in urls:
            content = extract_web_content(url)
            if content:
                writer.add_document(filename=url, content=preprocess(content))

    st.success("✅ Successfully indexed all files and URLs.")

# Query and search functionality
query = st.text_input("🔎 Enter search query here")

if query:
    ix = open_dir(INDEX_DIR)
    with ix.searcher() as searcher:
        # Configure the query parser with advanced features
        parser = QueryParser("content", ix.schema, group=OrGroup.factory(0.9))
        parsed_query = parser.parse(query)

        # Configure highlighting
        results = searcher.search(parsed_query, limit=20)
        results.formatter = highlight.HtmlFormatter(tagname="span", classname="highlight")

        st.subheader("📄 Results:")

        if results:
            filtered_results = []
            # Debug information
            st.write(f"Total results before filtering: {len(results)}")
            st.write(f"Selected file types: {selected_types}")
            
            for r in results:
                filename = r['filename'].lower()
                # Extract the base filename without the additional information
                base_filename = filename.split(' (')[0]
                file_ext = os.path.splitext(base_filename)[1].lower()
                file_type = None
                
                if file_ext == '.pdf':
                    file_type = 'PDF'
                elif file_ext == '.txt':
                    file_type = 'TXT'
                elif file_ext == '.csv':
                    file_type = 'CSV'
                elif file_ext == '.xlsx':
                    file_type = 'Excel'
                elif file_ext == '.json':
                    file_type = 'JSON'
                elif filename.startswith('http'):
                    file_type = 'Web'
                
                # Debug information for each file
                st.write(f"Processing file: {filename}, Base filename: {base_filename}, Type: {file_type}")
                
                if file_type in selected_types:
                    filtered_results.append(r)

            st.write(f"Results after filtering: {len(filtered_results)}")

            if not filtered_results:
                st.warning("😕 No results match the filter criteria.")
                # Show all results without filtering for debugging
                st.write("Showing all results without filtering:")
                for r in results:
                    st.write(f"File: {r['filename']}, Score: {r.score}")
            else:
                # Group results by filename
                results_by_file = {}
                for r in filtered_results:
                    filename = r['filename']
                    if filename not in results_by_file:
                        results_by_file[filename] = []
                    results_by_file[filename].append(r)

                # Display results grouped by file
                for filename, file_results in results_by_file.items():
                    # Extract base filename for type detection
                    base_filename = filename.split(' (')[0]
                    file_ext = os.path.splitext(base_filename)[1].lower()
                    file_type = 'Web' if filename.startswith('http') else file_ext[1:].upper()
                    
                    st.markdown(f"### 📄 File: {filename} ({file_type})")
                    st.markdown(f"**Found in {len(file_results)} locations**")
                    
                    for r in file_results:
                        st.markdown(f"**Score:** {r.score:.2f}")
                        st.markdown(f"**Content:** {r.highlights('content')}", unsafe_allow_html=True)
                        st.markdown("---")
        else:
            st.warning("😕 No matching results found.")
