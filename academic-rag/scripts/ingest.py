import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF files into FAISS")
    parser.add_argument("--dir", default="data/papers", help="Directory containing PDFs")
    parser.add_argument("--file", default=None, help="Single PDF path")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--reindex", action="store_true", help="Re-process PDFs already in the index")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        count = pipeline.index_documents_from_pdf(str(pdf_path))
        print(f"Indexed {count} chunks from {pdf_path.name}")
        return

    pdf_dir = Path(args.dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    indexed_sources: set[str] = set()
    if not args.reindex:
        indexed_sources = {
            Path(str(doc.get("source_name") or "")).name
            for doc in pipeline.retriever.list_documents()
        }
        if indexed_sources:
            print(f"Skipping {len(indexed_sources)} already-indexed PDFs (use --reindex to force).")

    total = 0
    skipped = 0
    for pdf in pdf_files:
        if pdf.name in indexed_sources:
            print(f"[skip] {pdf.name}")
            skipped += 1
            continue
        count = pipeline.index_documents_from_pdf(str(pdf))
        total += count
        print(f"{pdf.name}: {count} chunks")

    print(f"Done. Indexed {total} new chunks, skipped {skipped} PDFs.")


if __name__ == "__main__":
    main()
