import ezdxf
from ezdxf import colors
from ezdxf.layouts import Modelspace
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime
import xml.etree.ElementTree as ET

class CADExporter:
    """Professional CAD export functionality for architectural drawings"""

    def __init__(self):
        self.supported_formats = ['dxf', 'svg', 'pdf', 'dwg']
        self.export_settings = {
            'line_weight': 0.25,
            'text_height': 2.5,
            'dimension_scale': 1.0,
            'layer_colors': {
                'walls': colors.BLACK,
                'doors': colors.YELLOW,
                'windows': colors.BLUE,
                'furniture': colors.GREEN,
                'dimensions': colors.RED,
                'text': colors.BLACK
            }
        }

    def export_to_dxf(self, zones: List[Dict], results: Dict, output_path: str, **kwargs) -> str:
        """Export analysis results to DXF format"""

        # Create new DXF document
        doc = ezdxf.new('R2010')
        doc.units = ezdxf.units.M  # Set units to meters

        # Get modelspace
        msp = doc.modelspace()

        # Create layers
        self._create_layers(doc)

        # Export zones (rooms)
        self._export_zones_to_dxf(msp, zones, results)

        # Export furniture/equipment placements
        if 'placements' in results:
            self._export_placements_to_dxf(msp, results['placements'])

        # Export room labels and dimensions
        self._export_annotations_to_dxf(msp, zones, results)

        # Add title block
        self._add_title_block_dxf(msp, results)

        # Save file
        doc.saveas(output_path)
        return output_path

    def export_to_svg(self, zones: List[Dict], results: Dict, output_path: str) -> str:
        """Export analysis results to SVG format"""

        # Calculate drawing bounds
        bounds = self._calculate_bounds(zones)
        width = bounds['max_x'] - bounds['min_x']
        height = bounds['max_y'] - bounds['min_y']

        # Create SVG content
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width * 100}" height="{height * 100}" viewBox="{bounds['min_x']} {bounds['min_y']} {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .room-boundary {{ fill: none; stroke: black; stroke-width: 0.1; }}
            .furniture {{ fill: rgba(0,128,0,0.3); stroke: green; stroke-width: 0.05; }}
            .room-label {{ font-family: Arial; font-size: 0.5; text-anchor: middle; }}
            .dimensions {{ stroke: red; stroke-width: 0.02; fill: none; }}
        </style>
    </defs>
'''

        # Export zones
        for i, zone in enumerate(zones):
            if 'points' in zone:
                points_str = ' '.join([f"{p[0]},{p[1]}" for p in zone['points']])
                svg_content += f'    <polygon points="{points_str}" class="room-boundary"/>\n'

                # Add room label
                centroid = self._calculate_centroid(zone['points'])
                room_type = results.get('room_analysis', {}).get(str(i), {}).get('room_type', 'Unknown')
                svg_content += f'    <text x="{centroid[0]}" y="{centroid[1]}" class="room-label">{room_type}</text>\n'

        # Export furniture placements
        if 'placements' in results:
            for placement in results['placements']:
                x, y = placement.get('position', {}).get('x', 0), placement.get('position', {}).get('y', 0)
                w, h = placement.get('dimensions', {}).get('width', 1), placement.get('dimensions', {}).get('height', 1)
                svg_content += f'    <rect x="{x-w/2}" y="{y-h/2}" width="{w}" height="{h}" class="furniture"/>\n'

        svg_content += '</svg>'

        # Save file
        with open(output_path, 'w') as f:
            f.write(svg_content)

        return output_path

    def export_to_pdf(self, zones: List[Dict], results: Dict, output_path: str) -> str:
        """Export analysis results to PDF format"""
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm

        c = canvas.Canvas(output_path, pagesize=A4)

        # Calculate scale to fit page
        bounds = self._calculate_bounds(zones)
        page_width, page_height = A4
        margin = 20 * mm

        drawing_width = page_width - 2 * margin
        drawing_height = page_height - 2 * margin - 60 * mm  # Space for title and legend

        zone_width = bounds['max_x'] - bounds['min_x']
        zone_height = bounds['max_y'] - bounds['min_y']

        scale_x = drawing_width / zone_width if zone_width > 0 else 1
        scale_y = drawing_height / zone_height if zone_height > 0 else 1
        scale = min(scale_x, scale_y) * 0.8  # 80% of available space

        # Transform coordinates
        def transform_point(x, y):
            tx = margin + (x - bounds['min_x']) * scale
            ty = page_height - margin - 40 * mm - (y - bounds['min_y']) * scale
            return tx, ty

        # Draw title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin, "Architectural Space Analysis Report")

        c.setFont("Helvetica", 10)
        c.drawString(margin, page_height - margin - 20, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Draw zones
        c.setStrokeColor('black')
        c.setLineWidth(1)

        for i, zone in enumerate(zones):
            if 'points' in zone and len(zone['points']) > 2:
                # Draw room boundary
                points = [transform_point(p[0], p[1]) for p in zone['points']]
                path = c.beginPath()
                path.moveTo(*points[0])
                for point in points[1:]:
                    path.lineTo(*point)
                path.close()
                c.drawPath(path)

                # Add room label
                centroid = self._calculate_centroid(zone['points'])
                tx, ty = transform_point(centroid[0], centroid[1])
                room_type = results.get('room_analysis', {}).get(str(i), {}).get('room_type', 'Unknown')
                area = zone.get('area', 0)

                c.setFont("Helvetica", 8)
                c.drawCentredText(tx, ty + 5, room_type)
                c.drawCentredText(tx, ty - 5, f"{area:.1f} m²")

        # Draw furniture placements
        if 'placements' in results:
            c.setFillColor('green')
            c.setStrokeColor('darkgreen')
            for placement in results['placements']:
                pos = placement.get('position', {})
                dims = placement.get('dimensions', {})
                if pos and dims:
                    x, y = pos.get('x', 0), pos.get('y', 0)
                    w, h = dims.get('width', 1), dims.get('height', 1)

                    # Transform furniture rectangle
                    x1, y1 = transform_point(x - w/2, y - h/2)
                    x2, y2 = transform_point(x + w/2, y + h/2)

                    c.rect(x1, y2, x2-x1, y1-y2, fill=1, stroke=1)

        # Add statistics
        stats_y = 100
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, stats_y, "Analysis Summary:")

        c.setFont("Helvetica", 10)
        total_area = sum(zone.get('area', 0) for zone in zones)
        total_boxes = results.get('total_boxes', 0)
        efficiency = results.get('optimization', {}).get('total_efficiency', 0) * 100

        c.drawString(margin, stats_y - 20, f"Total Area: {total_area:.1f} m²")
        c.drawString(margin, stats_y - 35, f"Equipment Placed: {total_boxes}")
        c.drawString(margin, stats_y - 50, f"Space Efficiency: {efficiency:.1f}%")

        c.save()
        return output_path

    def create_export_package(self, zones: List[Dict], results: Dict, 
                            formats: List[str] = None) -> str:
        """Create a comprehensive export package with multiple formats"""

        if formats is None:
            formats = ['dxf', 'svg', 'pdf']

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            package_files = []

            # Export in each requested format
            for fmt in formats:
                if fmt in self.supported_formats:
                    filename = f"architectural_analysis.{fmt}"
                    filepath = os.path.join(temp_dir, filename)

                    if fmt == 'dxf':
                        self.export_to_dxf(zones, results, filepath)
                    elif fmt == 'svg':
                        self.export_to_svg(zones, results, filepath)
                    elif fmt == 'pdf':
                        self.export_to_pdf(zones, results, filepath)

                    package_files.append(filepath)

            # Add analysis report
            report_path = os.path.join(temp_dir, "analysis_report.json")
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            package_files.append(report_path)

            # Create zip package
            package_path = "architectural_analysis_package.zip"
            with zipfile.ZipFile(package_path, 'w') as zipf:
                for file_path in package_files:
                    zipf.write(file_path, os.path.basename(file_path))

            return package_path

    def _create_layers(self, doc):
        """Create standard architectural layers"""
        layers = [
            ('WALLS', colors.BLACK),
            ('DOORS', colors.YELLOW),
            ('WINDOWS', colors.BLUE),
            ('FURNITURE', colors.GREEN),
            ('DIMENSIONS', colors.RED),
            ('TEXT', colors.BLACK),
            ('EQUIPMENT', colors.MAGENTA)
        ]

        for layer_name, color in layers:
            doc.layers.new(layer_name, dxfattribs={'color': color})

    def _export_zones_to_dxf(self, msp: Modelspace, zones: List[Dict], results: Dict):
        """Export room zones to DXF"""
        for i, zone in enumerate(zones):
            if 'points' in zone and len(zone['points']) > 2:
                # Create room boundary
                points_3d = [(p[0], p[1], 0) for p in zone['points']]
                msp.add_lwpolyline(points_3d, close=True, dxfattribs={'layer': 'WALLS'})

    def _export_placements_to_dxf(self, msp: Modelspace, placements):
        """Export equipment placements to DXF"""
        if isinstance(placements, dict):
            # Handle dictionary of zone placements
            for zone_name, zone_placements in placements.items():
                for placement in zone_placements:
                    self._export_single_placement(msp, placement)
        elif isinstance(placements, list):
            # Handle list of placements
            for placement in placements:
                self._export_single_placement(msp, placement)
    
    def _export_single_placement(self, msp: Modelspace, placement):
        """Export a single placement to DXF"""
        if isinstance(placement, dict):
            # Handle standard placement format
            pos = placement.get('position', [0, 0])
            size = placement.get('size', [1, 1])
            
            if isinstance(pos, list) and len(pos) >= 2:
                x, y = pos[0], pos[1]
            else:
                x, y = 0, 0
                
            if isinstance(size, list) and len(size) >= 2:
                w, h = size[0], size[1]
            else:
                w, h = 1, 1

                # Create rectangle for equipment
            points = [
                (x - w/2, y - h/2, 0),
                (x + w/2, y - h/2, 0),
                (x + w/2, y + h/2, 0),
                (x - w/2, y + h/2, 0)
            ]
            msp.add_lwpolyline(points, close=True, dxfattribs={'layer': 'EQUIPMENT'})

    def _export_annotations_to_dxf(self, msp: Modelspace, zones: List[Dict], results: Dict):
        """Export text annotations and dimensions to DXF"""
        for i, zone in enumerate(zones):
            if 'points' in zone:
                centroid = self._calculate_centroid(zone['points'])
                room_type = results.get('room_analysis', {}).get(str(i), {}).get('room_type', 'Unknown')
                area = zone.get('area', 0)

                # Add room type label
                text_entity = msp.add_text(
                    room_type,
                    dxfattribs={
                        'layer': 'TEXT',
                        'height': self.export_settings['text_height']
                    }
                )
                text_entity.dxf.insert = (centroid[0], centroid[1] + 1, 0)

                # Add area label
                area_entity = msp.add_text(
                    f"{area:.1f} m²",
                    dxfattribs={
                        'layer': 'TEXT',
                        'height': self.export_settings['text_height'] * 0.8
                    }
                )
                area_entity.dxf.insert = (centroid[0], centroid[1] - 1, 0)

    def _add_title_block_dxf(self, msp: Modelspace, results: Dict):
        """Add title block to DXF"""
        # Simple title block
        title_text = "Architectural Space Analysis"
        date_text = datetime.now().strftime("%Y-%m-%d")

        msp.add_text(
            title_text,
            dxfattribs={
                'layer': 'TEXT',
                'height': self.export_settings['text_height'] * 2
            }
        ).set_pos((0, -5, 0))

        msp.add_text(
            date_text,
            dxfattribs={
                'layer': 'TEXT',
                'height': self.export_settings['text_height']
            }
        ).set_pos((0, -8, 0))

    def _calculate_bounds(self, zones: List[Dict]) -> Dict:
        """Calculate bounding box for all zones"""
        all_points = []
        for zone in zones:
            if 'points' in zone:
                all_points.extend(zone['points'])

        if not all_points:
            return {'min_x': 0, 'min_y': 0, 'max_x': 10, 'max_y': 10}

        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]

        return {
            'min_x': min(x_coords),
            'min_y': min(y_coords),
            'max_x': max(x_coords),
            'max_y': max(y_coords)
        }

    def _calculate_centroid(self, points: List[tuple]) -> tuple:
        """Calculate centroid of polygon"""
        if not points:
            return (0, 0)

        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return (x, y)

    def generate_drawing_package(self, zones: List[Dict], analysis_results: Dict, 
                                  output_dir: str) -> Dict[str, str]:
        """Generate complete technical drawing package"""
        package_files = {}

        try:
            # Plan view DXF
            plan_path = os.path.join(output_dir, "architectural_plan.dxf")
            self.export_to_dxf(zones, analysis_results, plan_path)
            package_files['plan_dxf'] = plan_path

            # Preview SVG
            preview_path = os.path.join(output_dir, "plan_preview.svg")
            self.export_to_svg(zones, analysis_results, preview_path)
            package_files['preview_svg'] = preview_path

            # 3D Model (if available)
            model_path = os.path.join(output_dir, "3d_model.obj")
            try:
                self.export_to_3d_model(zones, analysis_results, model_path)
                package_files['3d_model'] = model_path
            except:
                pass  # 3D export is optional

        except Exception as e:
            print(f"Drawing package generation error: {e}")

        return package_files

    def create_technical_drawing_package(self, zones: List[Dict], analysis_results: Dict, 
                                       output_dir: str) -> Dict[str, str]:
        """Create comprehensive technical drawing package"""
        return self.generate_drawing_package(zones, analysis_results, output_dir)