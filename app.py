import os
import streamlit as st
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
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
    filtered_tokens = [w for w in tokens if w not in stop_words]
    pos_tags = pos_tag(filtered_tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return ' '.join(lemmatized).strip()


# Extraction functions
def extract_pdf_pages(file_path):
    reader = PdfReader(file_path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def extract_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


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
                writer.add_document(filename=uploaded_file.name, content=preprocess(text))
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
        parser = QueryParser("content", ix.schema)
        parsed_query = parser.parse(preprocess(query))
        results = searcher.search(parsed_query, limit=20)
        st.subheader("📄 Results:")

        if results:
            filtered_results = []
            for r in results:
                filename = r['filename'].lower()
                if any(ft.lower() in filename for ft in selected_types):
                    filtered_results.append(r)

            if not filtered_results:
                st.warning("😕 No results match the filter criteria.")
            else:
                seen = set()
                for r in filtered_results:
                    key = r['filename']
                    if key in seen:
                        continue
                    seen.add(key)
                    st.write(f"📌 **{r['filename']}** (Score: {r.score:.2f})")
                    st.markdown(f"🔍 Snippet: {r.highlights('content')}", unsafe_allow_html=True)
        else:
            st.warning("😕 No matching results found.")
