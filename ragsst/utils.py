from typing import List, Tuple
import os
from pypdf import PdfReader
import hashlib
from docx import Document


def list_files(
    path: str, walksubdirs: bool = True, extensions: str | Tuple[str] = ""
) -> List[str]:
    """Returns a List of strings with the file names on the given path"""

    if walksubdirs:
        files_list = [
            os.path.join(root, f)
            for root, dirs, files in os.walk(path)
            for f in files
            if f.endswith(extensions)
        ]
    else:
        files_list = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(extensions)
        ]

    return sorted(files_list)


def read_file(doc: str) -> str:
    """Get text from pdf and txt files"""
    text = ''
    if doc.endswith('.txt'):
        with open(doc, 'r') as f:
            text = f.read()
    elif doc.endswith('.pdf'):
        pdf_reader = PdfReader(doc)
        text = ''.join([page.extract_text() for page in pdf_reader.pages])
    elif doc.endswith('.docx'):
        docx_doc = Document(doc)
        text = '\n'.join([para.text for para in docx_doc.paragraphs])
    return text


def split_text_basic(text: str, max_words: int = 256) -> List[str]:
    """Split text in chunks with less than max_words"""

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines(True) if l.strip()]

    chunks = []
    chunk = ''
    for l in lines:
        if len(chunk.split() + l.split()) <= max_words:
            chunk += l  # if splitline(False) do += "\n" + l
            continue
        chunks.append(chunk)
        chunk = l

    if chunk:
        chunks.append(chunk)

    return chunks


def split_text(text: str, max_words: int = 256, max_title_words: int = 4) -> List[str]:
    """Split text in trivial context-awared chunks with less than max_words"""

    punctuations = ('.', '?', '!', '”', '"')

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines() if l.strip()]

    chunks = []
    chunk = []
    chunk_length = 0
    for l in lines:
        line_length = len(l.split())
        if chunk_length + line_length <= max_words and (
            line_length > max_title_words
            or l.strip().endswith(punctuations)
            or all(len(s.split()) <= max_title_words for s in chunk)
        ):
            chunk.append(l)
            chunk_length += line_length
            continue
        chunks.append('\n'.join(chunk))
        chunk = [l]
        chunk_length = len(l.split())

    if chunk:
        chunks.append('\n'.join(chunk))

    return chunks


def hash_file(filename: str, block_size: int = 128 * 64) -> str:

    h = hashlib.sha1()

    with open(filename, 'rb') as f:
        while data := f.read(block_size):
            h.update(data)

    return h.hexdigest()
