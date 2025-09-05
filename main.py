import os, io
from flask import Flask, request, jsonify
import fitz  # PyMuPDF

app = Flask(__name__)

def is_green(rgb):  # rgb attendu comme [r,g,b] 0..1
    if not rgb: return False
    r, g, b = rgb
    return (g >= 0.6) and (g >= r + 0.10) and (g >= b + 0.10)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/extract", methods=["POST"])
def extract():
    if request.content_type and "application/pdf" in request.content_type.lower():
        pdf_bytes = request.get_data()
    else:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No PDF provided"}), 400
        pdf_bytes = f.read()

    try:
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    except Exception as e:
        return jsonify({"error": f"Failed to open PDF: {e}"}), 400

    highlights = []
    for i in range(len(doc)):
        page = doc[i]
        annots = page.annots()
        if not annots: 
            continue
        for a in annots:
            atype = getattr(a, "type", (None, ""))  # (code, name)
            if atype and (atype[0] == 8 or "Highlight" in (atype[1] or "")):
                color = None
                try:
                    color = (a.colors or {}).get("stroke", None)
                except Exception:
                    pass
                texts = []
                quads = getattr(a, "vertices", None)
                if quads and len(quads) % 8 == 0:
                    for j in range(0, len(quads), 8):
                        q = quads[j:j+8]
                        xs, ys = q[0::2], q[1::2]
                        rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                        t = page.get_text("text", clip=rect).strip()
                        if t: texts.append(t)
                else:
                    t = page.get_text("text", clip=a.rect).strip()
                    if t: texts.append(t)
                if texts:
                    highlights.append({
                        "page": i + 1,
                        "text": " ".join(texts).strip(),
                        "color_rgb": color,
                        "is_green": is_green(color)
                    })
    return jsonify({"highlights": highlights, "count": len(highlights)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
