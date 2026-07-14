import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loguru import logger


@dataclass
class Document:
    content: str
    metadata: Dict
    chunk_id: str


@dataclass
class SpecialBlock:
    """A table or formula-dense chunk extracted during parsing, kept for LLM summarization."""
    raw_chunk_id: str        # same id as the corresponding Document.chunk_id
    chunk_type: str          # "table" | "formula"
    table_markdown: str      # the raw chunk content containing table/formula material
    caption: str             # Table N: ... line if found
    section_title: str       # nearest ## heading above the block
    prev_context: str        # ~500 chars before the block
    next_context: str        # ~500 chars after the block


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting symbols while preserving content and structure.
    Tables and math blocks are left intact so _split_into_segments can detect them."""
    # Remove heading markers but keep heading text as a line
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold / italic
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Unwrap markdown links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove bare citation markers like [1], [2,3], [Smith et al.]
    text = re.sub(r"\[\d[\d,\s]*\]", "", text)
    # Collapse runs of blank lines left by removed elements
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_with_marker(pdf_path: str, device: str = "cpu") -> tuple[str, str]:
    """Return (plain_text, paper_title). Lazy-imports Marker so the rest of
    the codebase doesn't pay the import cost when Marker isn't used."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    try:
        artifact_dict = create_model_dict(device=device)
    except TypeError:
        artifact_dict = create_model_dict()

    converter = PdfConverter(artifact_dict=artifact_dict)
    rendered = converter(pdf_path)
    markdown, _, _ = text_from_rendered(rendered)  # second value is ext string ("md"), not metadata

    title = ""
    first_heading = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if first_heading:
        title = first_heading.group(1).strip()

    # Strip markdown symbols before returning so chunks embed as clean prose
    cleaned = _strip_markdown(markdown)
    return cleaned, title


def _parse_with_pypdf2(pdf_path: str) -> tuple[str, str]:
    """Fallback: extract plain text with PyPDF2 when Marker is unavailable."""
    import PyPDF2

    full_text = ""
    with open(pdf_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        title = _extract_title_pypdf2(reader, fallback=Path(pdf_path).stem)
        for page in reader.pages:
            text = page.extract_text() or ""
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            full_text += text.strip() + "\n"
    return full_text, title


class PDFParser:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        device: str = "cpu",
        engine: str = "pypdf2",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.engine = engine.lower()
        if self.engine not in {"pypdf2", "marker"}:
            raise ValueError("PDF parser engine must be either 'pypdf2' or 'marker'")

    def parse_pdf(
        self, pdf_path: str, source_name: str = None
    ) -> tuple[List[Document], List[SpecialBlock]]:
        """Return (documents, special_blocks).

        special_blocks contains one SpecialBlock per table/formula chunk,
        carrying the raw content and context needed for LLM summarisation.
        The corresponding Document in documents already has type/raw_chunk_id
        set in its metadata so the pipeline can match them up.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        logger.info("Parsing PDF: {}", path.name)
        source = source_name or path.name

        if self.engine == "pypdf2":
            return self._parse_pdf_with_pypdf2(path, source)

        try:
            full_text, marker_title = _parse_with_marker(pdf_path, device=self.device)
            paper_title = marker_title or Path(source).stem
            logger.info("Marker parsing succeeded for {}", path.name)
        except Exception as exc:
            logger.warning("Marker unavailable ({}), falling back to PyPDF2", exc)
            return self._parse_pdf_with_pypdf2(path, source)

        # Marker path: preserve table/formula blocks and return SpecialBlock items
        # for summary-indexing.
        segments = self._split_into_segments(full_text)
        chunk_records = self._merge_segments(segments)
        chunk_records = self._apply_overlap_to_records(chunk_records)

        # Build documents and collect special blocks in one pass
        documents: List[Document] = []
        special_blocks: List[SpecialBlock] = []

        chunk_idx = 0
        for chunk_text, chunk_type in chunk_records:
            if not chunk_text:
                continue
            chunk_id = f"{path.stem}_chunk_{chunk_idx}"
            is_special = chunk_type in ("table", "formula")
            metadata: Dict = {
                "source": source,
                "paper_title": paper_title,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunk_records),
                "parser_engine": self.engine,
                "type": chunk_type,
            }
            if is_special:
                metadata["raw_chunk_id"] = chunk_id

            documents.append(Document(
                content=chunk_text,
                metadata=metadata,
                chunk_id=chunk_id,
            ))

            if is_special:
                block = _extract_special_block(
                    chunk_id=chunk_id,
                    table_text=chunk_text,
                    full_text=full_text,
                    chunk_type=chunk_type,
                )
                special_blocks.append(block)

            chunk_idx += 1

        logger.info(
            "Parsed {} chunks ({} special) from {}",
            len(documents), len(special_blocks), path.name,
        )
        return documents, special_blocks

    def _parse_pdf_with_pypdf2(
        self,
        path: Path,
        source: str,
    ) -> tuple[List[Document], List[SpecialBlock]]:
        """Legacy PyPDF2 path: plain text extraction, page map, direct splitting.

        This intentionally avoids marker-style special block detection so PyPDF2
        remains the stable baseline chunking strategy.
        """
        import PyPDF2

        full_text = ""
        page_map = []

        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            paper_title = _extract_title_pypdf2(reader, fallback=source)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                start = len(full_text)
                full_text += text + "\n"
                page_map.append((start, len(full_text), page_num + 1))

        chunks = self._split_text(full_text)
        documents: List[Document] = []
        for idx, chunk in enumerate(chunks):
            chunk_start = full_text.find(chunk[:50])
            page_num = self._find_page(chunk_start, page_map)
            documents.append(
                Document(
                    content=chunk,
                    metadata={
                        "source": source,
                        "paper_title": paper_title,
                        "page": page_num,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "parser_engine": "pypdf2",
                    },
                    chunk_id=f"{path.stem}_chunk_{idx}",
                )
            )

        logger.info("Parsed {} PyPDF2 chunks from {}", len(documents), path.name)
        return documents, []

    def parse_text(self, text: str, source_name: str = "manual_input") -> List[Document]:
        chunks = self._split_text(text)
        return [
            Document(
                content=chunk,
                metadata={
                    "source": source_name,
                    "paper_title": source_name,
                    "chunk_index": idx,
                },
                chunk_id=f"{source_name}_chunk_{idx}",
            )
            for idx, chunk in enumerate(chunks)
        ]

    def _split_into_segments(self, text: str) -> List[tuple[str, str]]:
        """Split text into (segment_text, chunk_type) pairs.
        chunk_type is one of: "normal" | "table" | "formula".
        Tables and block formulae are kept whole; plain text is further split."""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        # Match Markdown tables and $$ ... $$ block formulae, in document order
        special_pattern = re.compile(
            r"(?P<table>(?:\|[^\n]+\|\n)+(?:\|[-:| ]+\|\n)(?:\|[^\n]+\|\n)*)"
            r"|(?P<formula>\$\$[\s\S]+?\$\$)",
            re.MULTILINE,
        )
        raw_segments: List[tuple[str, str]] = []
        last = 0
        for m in special_pattern.finditer(text):
            if m.start() > last:
                raw_segments.append((text[last : m.start()], "normal"))
            chunk_type = "table" if m.lastgroup == "table" else "formula"
            raw_segments.append((m.group(), chunk_type))
            last = m.end()
        if last < len(text):
            raw_segments.append((text[last:], "normal"))

        result: List[tuple[str, str]] = []
        for segment, chunk_type in raw_segments:
            if chunk_type in ("table", "formula"):
                if segment.strip():
                    result.append((segment.strip(), chunk_type))
            else:
                for plain_chunk in self._split_plain(segment):
                    result.append((plain_chunk, "normal"))
        return result

    def _split_text(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        separators = ["\n\n", "\n", "。", "；", "，", ".", "!", "?", " ", ""]

        for sep in separators:
            if sep and sep not in text:
                continue
            if sep:
                parts = text.split(sep)
                current_chunk = ""
                for part in parts:
                    addition = part + sep
                    if len(current_chunk) + len(addition) <= self.chunk_size:
                        current_chunk += addition
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = addition
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                step = max(1, self.chunk_size - self.chunk_overlap)
                for idx in range(0, len(text), step):
                    chunks.append(text[idx : idx + self.chunk_size])
            break

        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for idx in range(1, len(chunks)):
                overlap_text = chunks[idx - 1][-self.chunk_overlap :]
                overlapped.append(overlap_text + chunks[idx])
            return overlapped

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_page(self, char_pos: int, page_map: List) -> int:
        for start, end, page in page_map:
            if start <= char_pos < end:
                return page
        return 1

    def _merge_segments(self, segments: List[tuple[str, str]]) -> List[tuple[str, str]]:
        """Pack marker-level segments back into retrieval-sized chunks.

        Marker often emits every displayed formula as its own Markdown block. If each
        block becomes a Document, formula-heavy papers explode into hundreds of tiny
        chunks and hundreds of summary calls. This keeps table/formula boundaries
        visible for summarization while restoring chunking to the configured text size.
        """
        records: List[tuple[str, str]] = []
        current_parts: List[str] = []
        current_types: List[str] = []
        current_len = 0

        def combined_type(types: List[str]) -> str:
            if "table" in types:
                return "table"
            if "formula" in types:
                return "formula"
            return "normal"

        def flush() -> None:
            nonlocal current_parts, current_types, current_len
            if not current_parts:
                return
            records.append(("\n\n".join(current_parts).strip(), combined_type(current_types)))
            current_parts = []
            current_types = []
            current_len = 0

        for raw_text, chunk_type in segments:
            text = raw_text.strip()
            if not text:
                continue

            separator_len = 2 if current_parts else 0
            would_exceed = current_parts and (
                current_len + separator_len + len(text) > self.chunk_size
            )
            if would_exceed:
                flush()

            current_parts.append(text)
            current_types.append(chunk_type)
            current_len += separator_len + len(text)

            if len(text) >= self.chunk_size:
                flush()

        flush()
        return records

    def _apply_overlap_to_records(
        self,
        records: List[tuple[str, str]],
    ) -> List[tuple[str, str]]:
        if self.chunk_overlap <= 0 or len(records) <= 1:
            return records

        overlapped = [records[0]]
        for i in range(1, len(records)):
            prev_text, prev_type = records[i - 1]
            text, chunk_type = records[i]
            if prev_type in ("table", "formula"):
                overlapped.append((text, chunk_type))
            else:
                overlapped.append((prev_text[-self.chunk_overlap :] + text, chunk_type))
        return overlapped

    def _split_plain(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", "。", "；", "，", ".", "!", "?", " ", ""]
        for sep in separators:
            if sep and sep not in text:
                continue
            if sep:
                parts = text.split(sep)
                chunks: List[str] = []
                current = ""
                for part in parts:
                    addition = part + sep
                    if len(current) + len(addition) <= self.chunk_size:
                        current += addition
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        current = addition
                if current.strip():
                    chunks.append(current.strip())
                return chunks
            else:
                step = max(1, self.chunk_size - self.chunk_overlap)
                return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        return [text]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # 表格 chunk 不往下一个 chunk 渗透 overlap，避免污染
            if "|" in prev and re.search(r"\|[-:| ]+\|", prev):
                overlapped.append(chunks[i])
            else:
                overlapped.append(prev[-self.chunk_overlap :] + chunks[i])
        return overlapped


# ---------------------------------------------------------------------------
# Title extraction helpers (used by PyPDF2 fallback path)
# ---------------------------------------------------------------------------

def _extract_title_pypdf2(reader, fallback: str) -> str:
    import PyPDF2  # noqa: F401 — only called from fallback path

    metadata_title = ""
    try:
        metadata_title = str(getattr(reader.metadata, "title", "") or "").strip()
    except Exception:
        pass

    if _looks_like_title(metadata_title):
        return unicodedata.normalize("NFKC", metadata_title)

    first_page_text = ""
    try:
        if reader.pages:
            first_page_text = reader.pages[0].extract_text() or ""
    except Exception:
        pass

    title = _extract_title_from_first_page(first_page_text)
    if title:
        return unicodedata.normalize("NFKC", title)
    return unicodedata.normalize("NFKC", Path(fallback).stem)


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", Path(value or "").stem).lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"[^a-z0-9一-鿿]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_paper_title(reader, fallback: str) -> str:
    """Kept for any callers that still pass a PyPDF2 reader directly."""
    return _extract_title_pypdf2(reader, fallback)


def _extract_title_from_first_page(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    abstract_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if re.fullmatch(r"(?i)\s*abstract\s*[:.\-]?\s*", line)
            or re.match(r"(?i)\s*abstract\s*[:.\-]\s+", line)
        ),
        min(len(lines), 14),
    )
    candidates = [
        line
        for line in lines[:abstract_idx]
        if _looks_like_title(line)
        and not re.search(
            r"(?i)(arxiv|proceedings|conference|workshop|university|@|copyright|doi)",
            line,
        )
    ]
    if not candidates:
        return ""

    joined = []
    for line in candidates[:3]:
        joined.append(line)
        title = " ".join(joined)
        if len(title) >= 30:
            return title[:240]
    return " ".join(joined)[:240]


def _looks_like_title(value: str) -> bool:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) < 12 or len(value) > 260:
        return False
    if re.search(r"@|https?://|doi\.org", value, re.I):
        return False
    return len(value.split()) >= 3


# ---------------------------------------------------------------------------
# Special block extraction (table / formula context harvesting)
# ---------------------------------------------------------------------------

_CONTEXT_WINDOW = 500  # chars of surrounding text to capture


def _extract_special_block(
    chunk_id: str,
    table_text: str,
    full_text: str,
    chunk_type: str = "table",
) -> SpecialBlock:
    """Locate table_text inside full_text and harvest section title,
    caption, and surrounding context."""
    pos = full_text.find(table_text[:80])
    start = pos if pos != -1 else 0
    end = start + len(table_text)

    # --- prev / next context ---
    prev_raw = full_text[max(0, start - _CONTEXT_WINDOW) : start].strip()
    next_raw = full_text[end : end + _CONTEXT_WINDOW].strip()

    # --- caption: line immediately before or after the table matching "Table N" ---
    caption = ""
    cap_pattern = re.compile(
        r"(?:^|\n)((?:Table|Tab\.?)\s*\d+[^.\n]{0,120})", re.IGNORECASE
    )
    # look in the 300 chars before and after
    search_zone = full_text[max(0, start - 300) : end + 300]
    cap_match = cap_pattern.search(search_zone)
    if cap_match:
        caption = cap_match.group(1).strip()

    # --- section title: last ## or # heading before the table ---
    section_title = ""
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    preceding = full_text[:start]
    heading_matches = list(heading_pattern.finditer(preceding))
    if heading_matches:
        section_title = heading_matches[-1].group(1).strip()

    return SpecialBlock(
        raw_chunk_id=chunk_id,
        chunk_type=chunk_type,
        table_markdown=table_text,
        caption=caption,
        section_title=section_title,
        prev_context=prev_raw,
        next_context=next_raw,
    )
