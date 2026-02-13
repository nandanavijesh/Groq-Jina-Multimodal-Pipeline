import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def extract_text_from_page(page):
    """Extracts text from a single PDF page, using OCR as a fallback.
    Returns the extracted text and a boolean indicating if OCR was used.
    """
    text = page.get_text()
    ocr_used = False
    
    # If text is empty or very short, try OCR
    if len(text.strip()) < 50:
        try:
            # Render page to image at a higher DPI for better OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Extract text from image
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                text = ocr_text
                ocr_used = True
        except Exception as e:
            print(f"OCR Warning on page {page.number}: {e}")
    
    return text, ocr_used