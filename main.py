# main.py
import io
import os
import logging
from math import isfinite
from typing import List, Optional, Tuple, Dict, Any

from flask import Flask, request, jsonify
import fitz  # PyMuPDF

app = Flask(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = app.logger

# ---------- Utils ----------
def is_green(color_rgb: Optional[Tuple[float, float, float]]) -> bool:
    if not color_rgb:
        return False
    r, g, b = color_rgb
    return (g >= 0.6) and (g >= r + 0.10) and (g >= b + 0.10)

def rect_from_quad_list(q: List[float]) -> fitz.Rect:
    xs = q[0::2]
    ys = q[1::2]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

def add_text_from_rect(page: fitz.Page, r: fitz.Rect, texts: List[str], bboxes: List[List[float]]):
    t = page.get_text("text", clip=r).strip()
    if t:
        texts.append(t)
        bboxes.append([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])

def _sanitize_json(x: Any) -> Any:
    """Rend tout sérialisable JSON (NaN/Inf→None, tuples→list, Rect→bbox)."""
    if isinstance(x, float):
        return x if isfinite(x) else None
    if isinstance(x, (int, str, type(None), bool)):
        return x
    if isinstance(x, tuple):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, list):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _sanitize_json(v) for k, v in x.items()}
    if hasattr(x, "x0") and hasattr(x, "y0") and hasattr(x, "x1") and hasattr(x, "y1"):
        try:
            return [float(x.x0), float(x.y0), float(x.x1), float(x.y1)]
        except Exception:
            return None
    return str(x)

def _rects_overlap(r1: fitz.Rect, r2: fitz.Rect, min_iou: float = 0.05) -> bool:
    inter = fitz.Rect(max(r1.x0, r2.x0), max(r1.y0, r2.y0), min(r1.x1, r2.x1), min(r1.y1, r2.y1))
    if inter.x1 <= inter.x0 or inter.y1 <= inter.y0:
        return False
    inter_area = (inter.x1 - inter.x0) * (inter.y1 - inter.y0)
    a1 = (r1.x1 - r1.x0) * (r1.y1 - r1.y0)
    a2 = (r2.x1 - r2.x0) * (r2.y1 - r2.y0)
    denom = max(a1 + a2 - inter_area, 1e-6)
    iou = inter_area / denom
    return iou >= min_iou

def _collect_text_markup_annots(page: fitz.Page):
    """Liste compacte d'annotations Text Markup → [{type, rect, color}]"""
    out = []
    try:
        annots = page.annots()
    except Exception:
        annots = None
    if not annots:
        return out

    for a in annots:
        try:
            name = (getattr(a, "typeString", "") or "").lower()
            if not any(k in name for k in ["highlight", "underline", "strike", "squiggly"]):
                continue
            color = None
            try:
                colors = getattr(a, "colors", None) or {}
                color = colors.get("stroke", None)  # stroke color pour Highlight
            except Exception:
                pass

            v = getattr(a, "vertices", None)
            rects: List[fitz.Rect] = []
            if v and hasattr(v, "__len__") and len(v) > 0:
                # 3 formats possibles: Quads, Points (4*n), liste plate (8*n)
                if hasattr(v[0], "rect"):
                    rects = [q.rect for q in v]
                elif hasattr(v[0], "x") and hasattr(v[0], "y"):
                    for i in range(0, len(v), 4):
                        if i + 3 < len(v):
                            q = fitz.Quad([v[i], v[i+1], v[i+2], v[i+3]])
                            rects.append(q.rect)
                elif isinstance(v[0], (int, float)):
                    for i in range(0, len(v), 8):
                        if i + 7 < len(v):
                            xs = v[i:i+8][0::2]; ys = v[i:i+8][1::2]
                            rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
            if not rects:
                rects = [a.rect]

            for r in rects:
                out.append({"type": name, "rect": r, "color": color})
        except Exception:
            continue
    return out

def _collect_visual_highlights(page: fitz.Page):
    """
    Détecte des remplissages vectoriels (rectangles / paths remplis).
    On utilise page.get_drawings(); sinon fallback bboxlog.
    Renvoie [{rect, fill}] (fill = (r,g,b) 0..1 ou None).
    """
    visual = []
    try:
        drawings = page.get_drawings()  # chemins / rects / fill / stroke …
        for d in drawings:
            fill = d.get("fill")
            if not fill:
                continue
            r = d.get("rect", None)
            if r:
                visual.append({"rect": r, "fill": fill})
                continue
            pts = []
            for it in d.get("items", []):
                ps = it[1] if len(it) > 1 else []
                for p in ps:
                    if hasattr(p, "x") and hasattr(p, "y"):
                        pts.append((p.x, p.y))
            if pts:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                visual.append({"rect": fitz.Rect(min(xs), min(ys), max(xs), max(ys)), "fill": fill})
    except Exception:
        # Fallback: opérations de remplissage (images / paths)
        try:
            for b in page.get_bboxlog():
                if str(b.get("type", "")).startswith("fill"):
                    bb = b.get("bbox")
                    if bb and len(bb) == 4:
                        visual.append({"rect": fitz.Rect(*bb), "fill": None})
        except Exception:
            pass
    return visual

def _spans_compacts(page: fitz.Page) -> List[Dict[str, Any]]:
    """Texte compact + marquages (annotations & visuels)."""
    spans: List[Dict[str, Any]] = []

    # 1) spans texte
    d = page.get_text("dict") or {}
    for bi, b in enumerate(d.get("blocks", []) or []):
        if int(b.get("type", 0)) != 0:   # 0 = texte ; 1 = image
            continue
        for li, l in enumerate(b.get("lines", []) or []):
            for si, s in enumerate(l.get("spans", []) or []):
                t = s.get("text") or ""
                if not t:
                    continue
                font = str(s.get("font") or "")
                flags = int(s.get("flags") or 0)
                is_bold = ("bold" in font.lower()) or (flags != 0)
                bbox = [float(x) for x in (s.get("bbox") or [])]
                color = s.get("color")
                try:
                    color_rgb = list(fitz.sRGB_to_rgb(int(color))) if isinstance(color, int) else None
                except Exception:
                    color_rgb = None
                spans.append({
                    "page": page.number + 1,
                    "block": bi, "line": li, "span": si,
                    "text": t, "bbox": bbox,
                    "font": font, "size": s.get("size"),
                    "is_bold": is_bold, "color_rgb": color_rgb
                })

    if not spans:
        return spans

    # 2) indices surlignage
    annot_rects = _collect_text_markup_annots(page)   # quads → rects + color
    visual_rects = _collect_visual_highlights(page)   # rectangles / paths remplis

    for s in spans:
        r = fitz.Rect(*s["bbox"])

        # a) annotations PDF (Text Markup)
        types, samples = set(), []
        for a in annot_rects:
            if _rects_overlap(r, a["rect"], min_iou=0.05):
                types.add(a["type"])
                samples.append({
                    "bbox": [float(a["rect"].x0), float(a["rect"].y0), float(a["rect"].x1), float(a["rect"].y1)],
                    "color_rgb": list(a["color"]) if a["color"] else None
                })
                if len(samples) >= 2:
                    break
        s["annot_mark"] = {"has": bool(types), "types": sorted(types), "samples": samples if types else []}

        # b) surlignages visuels (remplissages)
        vs = []
        for v in visual_rects:
            if _rects_overlap(r, v["rect"], min_iou=0.05):
                vs.append({
                    "bbox": [float(v["rect"].x0), float(v["rect"].y0), float(v["rect"].x1), float(v["rect"].y1)],
                    "fill_rgb": list(v["fill"]) if v.get("fill") else None
                })
                if len(vs) >= 2:
                    break
        s["visual_mark"] = {"has": bool(vs), "samples": vs}

    return spans

# ---------- Error handlers ----------
@app.errorhandler(400)
def handle_400(err):
    return jsonify({"error": "Bad Request", "detail": str(err)}), 400

@app.errorhandler(404)
def handle_404(err):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def handle_500(err):
    return jsonify({"error": "Internal Server Error"}), 500

# ---------- Routes ----------
@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "pdf-annotations", "version": "1.5.0"}), 200

@app.post("/parse")
def parse_pdf():
    """
    Extraction native (pas d'OCR).
      - compact=1             ⇒ spans compacts + marquages
      - pages=1-3,5           ⇒ filtre de pages
      - truncate_span=200     ⇒ coupe le texte des spans
      - debug=1               ⇒ renvoie l’erreur exacte (diagnostic)
    """
    import json

    q = request.args or {}
    compact = (q.get("compact") == "1")
    pages_param = (q.get("pages") or "").strip()
    trunc = q.get("truncate_span")
    truncate_span = int(trunc) if (trunc and str(trunc).isdigit()) else None
    debug = (q.get("debug") == "1")

    try:
        # Lire PDF
        if request.content_type and "application/pdf" in (request.content_type or "").lower():
            pdf_bytes = request.get_data()
        else:
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No PDF provided (send as application/pdf, or multipart with field 'file')"}), 400
            pdf_bytes = f.read()
        if not pdf_bytes:
            return jsonify({"error": "Empty request body"}), 400

        # Ouvrir PDF
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            logger.exception("fitz.open failed (/parse)")
            return jsonify({"error": f"Failed to open PDF: {e}"}), 400

        # Pages à traiter
        all_nums = list(range(1, len(doc) + 1))
        sel = set()
        if pages_param:
            for part in pages_param.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    try:
                        a, b = int(a), int(b)
                        for n in range(a, b + 1):
                            if 1 <= n <= len(doc):
                                sel.add(n)
                    except Exception:
                        pass
                else:
                    try:
                        n = int(part)
                        if 1 <= n <= len(doc):
                            sel.add(n)
                    except Exception:
                        pass
        page_numbers = sorted(sel) if sel else all_nums

        out_pages = []
        approx_bytes = 0
        BYTES_BUDGET = 28 * 1024 * 1024  # marge sous 32 MiB

        for pno in page_numbers:
            page = doc[pno - 1]

            if compact:
                spans = _spans_compacts(page)
                if truncate_span is not None:
                    for s in spans:
                        if isinstance(s.get("text"), str) and len(s["text"]) > truncate_span:
                            s["text"] = s["text"][:truncate_span]
                page_obj = {"number": pno, "spans": spans}
            else:
                # version complète: rawdict + annotations
                try:
                    raw = page.get_text("rawdict") or {}
                except Exception:
                    logger.exception("rawdict failed on page %s", pno)
                    raw = {}
                try:
                    for b in raw.get("blocks", []) or []:
                        for l in b.get("lines", []) or []:
                            for s in l.get("spans", []) or []:
                                font = str(s.get("font") or "").lower()
                                flags = int(s.get("flags") or 0)
                                s["is_bold"] = ("bold" in font) or (flags != 0)
                                if truncate_span is not None and isinstance(s.get("text"), str) and len(s["text"]) > truncate_span:
                                    s["text"] = s["text"][:truncate_span]
                except Exception:
                    logger.exception("post-process rawdict spans failed")

                annots_json = []
                try:
                    annots = page.annots()
                except Exception:
                    annots = None
                if annots:
                    for a in annots:
                        try:
                            name = (getattr(a, "typeString", "") or "").lower()
                            is_text_markup = any(k in name for k in ["highlight", "underline", "strike", "squiggly"])
                            color = None
                            try:
                                colors = getattr(a, "colors", None) or {}
                                color = colors.get("stroke", None)
                            except Exception:
                                pass

                            boxes, texts = [], []
                            if is_text_markup:
                                v = getattr(a, "vertices", None)

                                def add_rect(r: fitz.Rect):
                                    t = page.get_text("text", clip=r).strip()
                                    if t:
                                        if truncate_span is not None and len(t) > truncate_span:
                                            t = t[:truncate_span]
                                        texts.append(t)
                                        boxes.append([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])

                                try:
                                    if v:
                                        if hasattr(v, "__len__") and len(v) > 0 and hasattr(v[0], "rect"):
                                            for q in v:
                                                add_rect(q.rect)
                                        elif hasattr(v, "__len__") and len(v) > 0 and hasattr(v[0], "x") and hasattr(v[0], "y"):
                                            for i in range(0, len(v), 4):
                                                if i + 3 < len(v):
                                                    q = fitz.Quad([v[i], v[i+1], v[i+2], v[i+3]])
                                                    add_rect(q.rect)
                                        elif hasattr(v, "__len__") and len(v) >= 8 and isinstance(v[0], (int, float)):
                                            for i in range(0, len(v), 8):
                                                if i + 7 < len(v):
                                                    xs = v[i:i+8][0::2]; ys = v[i:i+8][1::2]
                                                    r = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                                                    add_rect(r)
                                    if not texts:
                                        add_rect(a.rect)
                                except Exception:
                                    try:
                                        add_rect(a.rect)
                                    except Exception:
                                        pass
                            else:
                                r = a.rect
                                boxes.append([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])

                            annots_json.append({
                                "type": name,
                                "color_rgb": list(color) if color else None,
                                "boxes": boxes,
                                "text": " ".join(texts).strip()
                            })
                        except Exception:
                            logger.exception("failed to process annotation on page %s", pno)

                page_obj = {"number": pno, "text_raw": raw, "annotations": annots_json}

            out_pages.append(page_obj)

            # Garde-fou taille
            try:
                approx_bytes += len(json.dumps(_sanitize_json(page_obj), ensure_ascii=False))
                if approx_bytes > BYTES_BUDGET:
                    break
            except Exception:
                pass

        resp = {
            "meta": {"page_count": len(doc), "returned_pages": len(out_pages), "compact": compact},
            "pages": out_pages
        }
        resp = _sanitize_json(resp)
        return jsonify(resp), 200

    except Exception as e:
        logger.exception("Unhandled error in /parse")
        if (request.args or {}).get("debug") == "1":
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        return jsonify({"error": "Internal Server Error"}), 500

@app.post("/extract")
def extract():
    """Highlights uniquement (texte sous quads + couleur + bboxes + is_green)."""
    try:
        if request.content_type and "application/pdf" in (request.content_type or "").lower():
            pdf_bytes = request.get_data()
        else:
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No PDF provided (send as application/pdf, or multipart with field 'file')"}), 400
            pdf_bytes = f.read()
        if not pdf_bytes:
            return jsonify({"error": "Empty request body"}), 400

        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            logger.exception("fitz.open failed")
            return jsonify({"error": f"Failed to open PDF: {e}"}), 400

        results: List[Dict[str, Any]] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            annots = page.annots()
            if not annots:
                continue
            for annot in annots:
                a_type = getattr(annot, "type", None)
                name = getattr(annot, "typeString", "") or ""
                code = a_type if isinstance(a_type, int) else (a_type[0] if (a_type and len(a_type) > 0) else None)
                if not ((code == 8) or ("Highlight" in name)):
                    continue
                color = None
                try:
                    colors = getattr(annot, "colors", None) or {}
                    color = colors.get("stroke", None)
                except Exception:
                    pass
                texts: List[str] = []; bboxes: List[List[float]] = []
                v = getattr(annot, "vertices", None)
                try:
                    if v:
                        if hasattr(v[0], "rect"):
                            for q in v:
                                add_text_from_rect(page, q.rect, texts, bboxes)
                        elif hasattr(v[0], "x") and hasattr(v[0], "y"):
                            for i in range(0, len(v), 4):
                                if i + 3 < len(v):
                                    q = fitz.Quad([v[i], v[i+1], v[i+2], v[i+3]])
                                    add_text_from_rect(page, q.rect, texts, bboxes)
                        elif isinstance(v[0], (int, float)):
                            for i in range(0, len(v), 8):
                                if i + 7 < len(v):
                                    r = rect_from_quad_list(v[i:i+8])
                                    add_text_from_rect(page, r, texts, bboxes)
                    if not texts:
                        add_text_from_rect(page, annot.rect, texts, bboxes)
                except Exception:
                    try:
                        add_text_from_rect(page, annot.rect, texts, bboxes)
                    except Exception:
                        pass
                if texts:
                    results.append({
                        "page": page_index + 1,
                        "text": " ".join(texts).strip(),
                        "color_rgb": list(color) if color else None,
                        "is_green": is_green(color),
                        "boxes": bboxes,
                    })
        return jsonify({"highlights": results, "count": len(results)}), 200

    except Exception as e:
        logger.exception("Unhandled error in /extract")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.post("/fulltext")
def fulltext():
    """Texte brut par page (diagnostic)."""
    try:
        if request.content_type and "application/pdf" in (request.content_type or "").lower():
            pdf_bytes = request.get_data()
        else:
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No PDF provided (send as application/pdf, or multipart with field 'file')"}), 400
            pdf_bytes = f.read()
        if not pdf_bytes:
            return jsonify({"error": "Empty request body"}), 400

        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            logger.exception("fitz.open failed (/fulltext)")
            return jsonify({"error": f"Failed to open PDF: {e}"}), 400

        pages = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")
            pages.append({"page": i + 1, "text": text})

        return jsonify({"pages": pages, "page_count": len(pages)}), 200

    except Exception as e:
        logger.exception("Unhandled error in /fulltext")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
