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
    """
    Heuristique simple : le canal G doit dominer.
    color_rgb: ex. (r, g, b) chacun 0..1 (PyMuPDF)
    """
    if not color_rgb:
        return False
    r, g, b = color_rgb
    return (g >= 0.6) and (g >= r + 0.10) and (g >= b + 0.10)


def rect_from_quad(quad: List[float]) -> fitz.Rect:
    """quad = [x0,y0,x1,y1,x2,y2,x3,y3] → rect englobant"""
    xs = quad[0::2]
    ys = quad[1::2]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def bbox_tuple(rect: fitz.Rect) -> List[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


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
    return jsonify({"status": "ok", "service": "pdf-annotations", "version": "1.0.1"}), 200


@app.post("/extract")
def extract():
    """
    Reçoit un PDF :
    - soit en binaire pur (Content-Type: application/pdf)
    - soit en multipart/form-data (champ 'file')
    Renvoie un JSON avec toutes les annotations 'Highlight' trouvées.
    """
    try:
        # ---- Lire les octets du PDF ----
        if request.content_type and "application/pdf" in request.content_type.lower():
            pdf_bytes = request.get_data()
        else:
            f = request.files.get("file")
            if not f:
                return jsonify({"error": "No PDF provided (send as application/pdf, or multipart with field 'file')"}), 400
            pdf_bytes = f.read()

        if not pdf_bytes:
            return jsonify({"error": "Empty request body"}), 400

        # ---- Ouvrir le PDF avec PyMuPDF ----
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            logger.exception("fitz.open failed")
            return jsonify({"error": f"Failed to open PDF: {e}"}), 400

        results: List[Dict[str, Any]] = []

        # ---- Parcourir pages & annotations ----
        for page_index in range(len(doc)):
            page = doc[page_index]
            annots = page.annots()
            if not annots:
                continue

            for annot in annots:
                # Compat nouvelles / anciennes versions :
                # - nouvelles : annot.type -> int (code)
                # - anciennes : annot.type -> (code, name)
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

                # 8 = Highlight (ou name contient "Highlight")
                is_highlight = (code == 8) or ("Highlight" in name)
                if not is_highlight:
                    continue

                # Couleur : stroke color (PyMuPDF stocke la couleur de surlignage là)
                color = None
                try:
                    colors = getattr(annot, "colors", None) or {}
                    color = colors.get("stroke", None)
                except Exception:
                    pass

                # Récupérer texte sous les quads / bbox
                texts: List[str] = []
                bboxes: List[List[float]] = []

                quads = getattr(annot, "vertices", None)
                if quads and len(quads) % 8 == 0:
                    for i in range(0, len(quads), 8):
                        rect = rect_from_quad(quads[i:i+8])
                        t = page.get_text("text", clip=rect).strip()
                        if t:
                            texts.append(t)
                            bboxes.append(bbox_tuple(rect))
                else:
                    try:
                        rect = annot.rect
                        t = page.get_text("text", clip=rect).strip()
                        if t:
                            texts.append(t)
                            bboxes.append(bbox_tuple(rect))
                    except Exception:
                        logger.exception("Failed reading text for annotation")

                if not texts:
                    continue

                results.append({
                    "page": page_index + 1,
                    "text": " ".join(texts).strip(),
                    "color_rgb": list(color) if color else None,  # [r,g,b] 0..1
                    "is_green": is_green(color),
                    "boxes": bboxes,
                })

        return jsonify({"highlights": results, "count": len(results)}), 200

    except Exception as e:
        logger.exception("Unhandled error in /extract")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.post("/fulltext")
def fulltext():
    """
    (Optionnel) Retourne le texte complet de chaque page (PDF natif).
    Utile pour audit / diagnostics ; n’effectue pas d’OCR.
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
    # Buildpacks / Cloud Run : écoute sur $PORT (8080 par défaut)
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
