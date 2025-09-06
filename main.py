import io
import os
import logging
from typing import List, Optional, Tuple, Dict, Any

from flask import Flask, request, jsonify
import fitz  # PyMuPDF

# ---------------------------
# App & Logging
# ---------------------------
app = Flask(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = app.logger

# ---------------------------
# Utils
# ---------------------------
def is_green(color_rgb: Optional[Tuple[float, float, float]]) -> bool:
    """Heuristique simple : canal G dominant (valeurs 0..1)."""
    if not color_rgb:
        return False
    r, g, b = color_rgb
    return (g >= 0.6) and (g >= r + 0.10) and (g >= b + 0.10)

def rect_from_quad_list(q: List[float]) -> fitz.Rect:
    """q = [x0,y0,x1,y1,x2,y2,x3,y3] → rect englobant"""
    xs = q[0::2]
    ys = q[1::2]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

def add_text_from_rect(page: fitz.Page, r: fitz.Rect, texts: List[str], bboxes: List[List[float]]):
    t = page.get_text("text", clip=r).strip()
    if t:
        texts.append(t)
        bboxes.append([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])

# ---------------------------
# Error Handlers → JSON
# ---------------------------
@app.errorhandler(400)
def handle_400(err):
    return jsonify({"error": "Bad Request", "detail": str(err)}), 400

@app.errorhandler(404)
def handle_404(err):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def handle_500(err):
    return jsonify({"error": "Internal Server Error"}), 500

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "pdf-annotations", "version": "1.2.0"}), 200

@app.post("/parse")
def parse_pdf():
    """
    Extraction *complète* (PDF natif, pas d'OCR) avec garde-fous:
    - pages=ex. "1-3,5" pour limiter les pages
    - truncate_span=ex. 200 pour couper le texte des spans (évite >32MiB)
    - debug=1 pour renvoyer l'exception dans la réponse JSON (diagnostic)
    """
    import json

    # --- params facultatifs ---
    q = request.args or {}
    pages_param = (q.get("pages") or "").strip()
    trunc = q.get("truncate_span")
    truncate_span = int(trunc) if (trunc and str(trunc).isdigit()) else None
    debug = (q.get("debug") == "1")

    try:
        # 1) Lire PDF (binaire pur ou multipart 'file')
        if request.content_type and "application/pdf" in request.content_type.lower():
            pdf_bytes = request.get_data()
        else:
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No PDF provided (send as application/pdf, or multipart with field 'file')"}), 400
            pdf_bytes = f.read()
        if not pdf_bytes:
            return jsonify({"error": "Empty request body"}), 400

        # 2) Ouvrir PDF
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            logger.exception("fitz.open failed (/parse)")
            return jsonify({"error": f"Failed to open PDF: {e}"}), 400

        # 3) Calcul des pages à traiter
        all_nums = list(range(1, len(doc) + 1))
        sel_nums = set()
        if pages_param:
            for part in pages_param.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    try:
                        a, b = int(a), int(b)
                        for n in range(a, b + 1):
                            if 1 <= n <= len(doc):
                                sel_nums.add(n)
                    except Exception:
                        pass
                else:
                    try:
                        n = int(part)
                        if 1 <= n <= len(doc):
                            sel_nums.add(n)
                    except Exception:
                        pass
        page_numbers = sorted(sel_nums) if sel_nums else all_nums

        out_pages = []
        approx_bytes = 0  # garde-fou pour rester < ~28MiB (marge)
        BYTES_BUDGET = 28 * 1024 * 1024

        for pno in page_numbers:
            page = doc[pno - 1]

            # 4) rawdict (robuste)
            try:
                raw = page.get_text("rawdict") or {}
            except Exception:
                logger.exception("rawdict failed on page %s", pno)
                raw = {}

            # 4a) is_bold + éventuel tronquage de span
            try:
                for b in raw.get("blocks", []) or []:
                    for l in b.get("lines", []) or []:
                        for s in l.get("spans", []) or []:
                            font = str(s.get("font") or "").lower()
                            flags = int(s.get("flags") or 0)
                            s["is_bold"] = ("bold" in font) or (flags != 0)
                            if truncate_span is not None:
                                t = s.get("text")
                                if isinstance(t, str) and len(t) > truncate_span:
                                    s["text"] = t[:truncate_span]
            except Exception:
                logger.exception("post-process rawdict spans failed")

            # 5) Annotations text-markup (robuste)
            annots_json = []
            try:
                annots = page.annots()
            except Exception:
                annots = None

            if annots:
                for a in annots:
                    try:
                        name = (getattr(a, "typeString", "") or "").lower()
                        if not any(k in name for k in ["highlight", "underline", "strike", "squiggly"]):
                            continue
                        color = None
                        try:
                            colors = getattr(a, "colors", None) or {}
                            color = colors.get("stroke", None)  # (r,g,b) 0..1 si dispo
                        except Exception:
                            pass

                        boxes, texts = [], []
                        v = getattr(a, "vertices", None)

                        def add_rect(r: fitz.Rect):
                            try:
                                t = page.get_text("text", clip=r).strip()
                            except Exception:
                                t = ""
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

                        annots_json.append({
                            "type": name,
                            "color_rgb": list(color) if color else None,
                            "boxes": boxes,
                            "text": " ".join(texts).strip()
                        })
                    except Exception:
                        logger.exception("failed to process annotation on page %s", pno)

            page_obj = {
                "number": pno,
                "size": [float(page.rect.width), float(page.rect.height)],
                "text_raw": raw,
                "annotations": annots_json
            }
            out_pages.append(page_obj)

            # 6) garde-fou taille réponse (~28MiB)
            try:
                approx_bytes += len(json.dumps(page_obj, ensure_ascii=False))
                if approx_bytes > BYTES_BUDGET:
                    break
            except Exception:
                pass

        resp = {"meta": {"page_count": len(doc), "returned_pages": len(out_pages)}, "pages": out_pages}
        return jsonify(resp), 200

    except Exception as e:
        logger.exception("Unhandled error in /parse")
        if (request.args or {}).get("debug") == "1":
            # renvoie le message d'erreur pour faciliter le debug côté Make
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        return jsonify({"error": "Internal Server Error"}), 500


@app.post("/extract")
def extract():
    """
    PDF -> JSON des surlignages "Highlight" uniquement :
    - binaire pur (Content-Type: application/pdf) OU multipart/form-data (champ 'file')
    - renvoie texte sous les quads + couleur + bboxes + is_green
    """
    try:
        if request.content_type and "application/pdf" in request.content_type.lower():
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
                # type highlight (compat int / tuple)
                a_type = getattr(annot, "type", None)
                name = getattr(annot, "typeString", "") or ""
                code = None
                if isinstance(a_type, int):
                    code = a_type
                else:
                    try:
                        code = a_type[0]
                        if not name:
                            name = (a_type[1] or "")
                    except Exception:
                        code = None

                is_highlight = (code == 8) or ("Highlight" in name)
                if not is_highlight:
                    continue

                # couleur = stroke color
                color = None
                try:
                    colors = getattr(annot, "colors", None) or {}
                    color = colors.get("stroke", None)
                except Exception:
                    pass

                # texte & bboxes depuis vertices (robuste)
                texts: List[str] = []
                bboxes: List[List[float]] = []
                v = getattr(annot, "vertices", None)
                try:
                    if v:
                        if hasattr(v[0], "rect"):  # fitz.Quad
                            for q in v:
                                add_text_from_rect(page, q.rect, texts, bboxes)
                        elif hasattr(v[0], "x") and hasattr(v[0], "y"):  # Points
                            for i in range(0, len(v), 4):
                                if i + 3 < len(v):
                                    q = fitz.Quad([v[i], v[i+1], v[i+2], v[i+3]])
                                    add_text_from_rect(page, q.rect, texts, bboxes)
                        elif isinstance(v[0], (int, float)):  # 8*n
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

                if not texts:
                    continue

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
        if request.content_type and "application/pdf" in request.content_type.lower():
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

# ---------------------------
# Entrée Cloud Run / Buildpacks
# ---------------------------
if __name__ == "__main__":
    # Buildpacks / Cloud Run écoutent sur $PORT (8080 par défaut)
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
