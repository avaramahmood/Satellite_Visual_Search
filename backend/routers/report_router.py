"""
Report & VLM Chat router.
  POST /api/report/generate — generate ML evaluation report (markdown) for a session
  POST /api/report/chat     — VLM Q&A grounded in search context
  GET  /api/report/metrics  — return pre-computed metrics for a session
  POST /api/report/pdf      — generate + return PDF evaluation report
"""
from __future__ import annotations

import io
import logging
import re
import textwrap

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, Response

from services.vlm_service import generate_evaluation_report, vlm_chat
from services.context_manager import load_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/report", tags=["Reporting & VLM Chat"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _markdown_to_pdf_bytes(markdown_text: str, session_id: str) -> bytes:
    """Convert a markdown string to a simple PDF using fpdf2."""
    from fpdf import FPDF  # lazy import — only needed for PDF endpoint

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # Title block
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Satellite Visual Search", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 8, "ML Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Session: {session_id}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    def _safe(text: str) -> str:
        """Replace characters outside latin-1 range with closest ASCII equivalent."""
        return (
            text
            .replace("\u2022", "-")   # bullet
            .replace("\u2013", "-")   # en-dash
            .replace("\u2014", "--")  # em-dash
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2026", "...")  # ellipsis
            .encode("latin-1", errors="replace").decode("latin-1")
        )

    # Render markdown line-by-line
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        # Strip inline markdown: **bold**, *italic*, `code`, [text](url)
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        clean = re.sub(r"\*(.+?)\*",     r"\1", clean)
        clean = re.sub(r"`(.+?)`",       r"\1", clean)
        clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", clean)
        clean = clean.strip()

        if not clean:
            pdf.ln(3)
            continue

        clean = _safe(clean)

        # Heading levels
        if clean.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(230, 230, 240)
            pdf.cell(0, 7, clean[4:], new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(1)
        elif clean.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, clean[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(1)
        elif clean.startswith("# "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 15)
            pdf.cell(0, 10, clean[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(2)
        elif clean.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 10)
            bullet_text = "  -  " + clean[2:]
            for chunk in textwrap.wrap(bullet_text, width=90):
                pdf.cell(0, 6, chunk, new_x="LMARGIN", new_y="NEXT")
        elif re.match(r"^\d+\.\s", clean):
            pdf.set_font("Helvetica", "", 10)
            for chunk in textwrap.wrap(clean, width=90):
                pdf.cell(0, 6, chunk, new_x="LMARGIN", new_y="NEXT")
        elif re.match(r"^\|.+\|$", clean):
            pdf.set_font("Courier", "", 8)
            cells = [c.strip() for c in clean.strip("|").split("|")]
            row_text = "  |  ".join(cells)
            pdf.cell(0, 5, row_text[:120], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
        elif re.match(r"^[-|]+$", clean):
            pdf.set_draw_color(180, 180, 180)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(1)
            pdf.set_draw_color(0, 0, 0)
        else:
            pdf.set_font("Helvetica", "", 10)
            for chunk in textwrap.wrap(clean, width=95):
                pdf.cell(0, 6, chunk, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# ── routes ────────────────────────────────────────────────────────────────────

@router.post("/generate")
async def generate_report(
    session_id: str = Form(...),
    ground_truth_count: int = Form(None),
):
    """
    Generates a Markdown evaluation report (Precision@K, mAP, NDCG, F1, etc.)
    for the k-NN results stored under the given session_id.
    """
    try:
        report_markdown = await generate_evaluation_report(session_id, ground_truth_count)
        logger.info("Evaluation report generated for session %s", session_id)
        return {"session_id": session_id, "report": report_markdown}
    except Exception as exc:
        logger.exception("Report generation failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat")
async def vlm_chat_endpoint(
    session_id: str = Form(...),
    message: str = Form(...),
):
    """VLM Q&A grounded in the k-NN context stored under session_id."""
    if not message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        response_text = await vlm_chat(session_id, message)
        return {"response": response_text}
    except Exception as exc:
        logger.exception("VLM chat failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/metrics")
def get_metrics(session_id: str):
    """Return pre-computed retrieval metrics for a session (no LLM needed)."""
    data = load_metrics(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(data)


@router.post("/pdf")
async def download_report_pdf(
    session_id: str = Form(...),
    ground_truth_count: int = Form(None),
):
    """Generate a PDF evaluation report and return it as a file download."""
    try:
        report_md = await generate_evaluation_report(session_id, ground_truth_count)
        pdf_bytes = _markdown_to_pdf_bytes(report_md, session_id)
        filename  = f"report_{session_id[:8]}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.exception("PDF report failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))
