import fitz  # PyMuPDF
from typing import List, Dict, Any
from shapely.geometry import Polygon
import logging

class PDFParser:
    """
    Parser for PDF floor plans to extract room geometry and dimensions.
    Uses PyMuPDF to extract vector paths and convert to polygons.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parse PDF file and extract zones (closed polygons)

        Args:
            file_bytes: Raw file content as bytes
            filename: Original filename

        Returns:
            List of zone dictionaries with points and metadata
        """
        zones = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                shapes = page.get_drawings()
                for shape in shapes:
                    points = []
                    for item in shape["items"]:
                        if item[0] == "l":  # line segment
                            points.append((item[1], item[2]))
                        elif item[0] == "re":  # rectangle
                            x, y, w, h = item[1], item[2], item[3], item[4]
                            points.extend([
                                (x, y),
                                (x + w, y),
                                (x + w, y + h),
                                (x, y + h)
                            ])
                    if points:
                        # Attempt to create polygon and check if closed
                        try:
                            poly = Polygon(points)
                            if poly.is_valid and poly.is_closed and poly.area > 0:
                                zones.append({
                                    "points": points,
                                    "layer": f"Page_{page_num}",
                                    "entity_type": "PDF_SHAPE",
                                    "closed": True,
                                    "area": poly.area
                                })
                        except Exception as e:
                            self.logger.warning(f"Invalid polygon in PDF parsing: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing PDF file {filename}: {e}")
            raise e

        return zones
