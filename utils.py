import os
from pypdf import PdfReader
import hashlib


def list_files(path, walksubdirs=True, extensions=''):
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


def read_file(doc):
    """Get text from pdf and txt files"""
    text = ''
    if doc.endswith('.txt'):
        with open(doc, 'r') as f:
            text = f.read()
    elif doc.endswith('.pdf'):
        pdf_reader = PdfReader(doc)
        text = ''.join([page.extract_text() for page in pdf_reader.pages])
    return text


def split_text_basic(text, max_words=256):
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


def split_text(text, max_words=256, max_title_words=4):
    """Split text in trivial context-awared chunks with less than max_words"""

    # List of lines skipping empty lines
    lines = [l for l in text.splitlines(True) if l.strip()]

    chunks = []
    chunk = []
    chunk_length = 0
    for l in lines:
        line_length = len(l.split())
        if chunk_length + line_length <= max_words and (
            line_length > max_title_words or all(len(s.split()) <= max_title_words for s in chunk)
        ):
            chunk.append(l)
            chunk_length += line_length
            continue
        chunks.append(''.join(chunk))  # if splitlines(False) do "\n".join()
        chunk = [l]
        chunk_length = len(l.split())

    if chunk:
        chunks.append(''.join(chunk))

    return chunks


def hash_file(filename, block_size=128 * 64):

    h = hashlib.sha1()

    with open(filename, 'rb') as f:
        while data := f.read(block_size):
            h.update(data)

    return h.hexdigest()


if __name__ == '__main__':
    # check chunk splitting
    docs_path = 'sample_data'
    files = list_files(docs_path, extensions=('.txt', '.pdf'))
    print("Found files:")
    print("\n".join(files))
    chosen_file = files[1]
    print(f"\nReading and splitting {chosen_file}...")
    text = read_file(chosen_file)
    chunks = split_text(text)
    print("\nChunks:")
    for c in chunks:
        print(c)
        print("Words Length:", len(c.split()))
        print(80 * '-')
