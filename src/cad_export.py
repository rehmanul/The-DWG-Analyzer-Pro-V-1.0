import ezdxf
from ezdxf import colors
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from io import BytesIO
import base64

class CADExporter:
    """Professional CAD export functionality for architectural drawings"""
    
    def __init__(self):
        self.dxf_version = 'R2018'
        self.units = 'Meters'
        self.precision = 3
        self.layer_standards = self._initialize_layer_standards()
        self.color_scheme = self._initialize_color_scheme()
        self.line_weights = self._initialize_line_weights()
    
    def _initialize_layer_standards(self) -> Dict[str, Dict]:
        """Initialize AIA layer standards for architectural drawings"""
        return {
            'A-WALL': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 0.5, 'description': 'Walls'},
            'A-DOOR': {'color': 3, 'linetype': 'CONTINUOUS', 'lineweight': 0.35, 'description': 'Doors'},
            'A-GLAZ': {'color': 4, 'linetype': 'CONTINUOUS', 'lineweight': 0.25, 'description': 'Windows/Glazing'},
            'A-FURN': {'color': 8, 'linetype': 'CONTINUOUS', 'lineweight': 0.18, 'description': 'Furniture'},
            'A-AREA': {'color': 2, 'linetype': 'DASHED', 'lineweight': 0.13, 'description': 'Area boundaries'},
            'A-ANNO-TEXT': {'color': 1, 'linetype': 'CONTINUOUS', 'lineweight': 0.13, 'description': 'Annotations'},
            'A-ANNO-DIMS': {'color': 1, 'linetype': 'CONTINUOUS', 'lineweight': 0.13, 'description': 'Dimensions'},
            'A-GRID': {'color': 6, 'linetype': 'CENTER', 'lineweight': 0.35, 'description': 'Grid lines'},
            'A-COLS': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 0.5, 'description': 'Columns'},
            'A-ROOF': {'color': 5, 'linetype': 'CONTINUOUS', 'lineweight': 0.35, 'description': 'Roof elements'}
        }
    
    def _initialize_color_scheme(self) -> Dict[str, int]:
        """Initialize color scheme for different room types"""
        return {
            'Office': colors.CYAN,
            'Conference Room': colors.BLUE,
            'Open Office': colors.MAGENTA,
            'Kitchen': colors.YELLOW,
            'Bathroom': colors.GREEN,
            'Storage': colors.BROWN,
            'Corridor': colors.WHITE,
            'Reception': colors.RED,
            'Lobby': colors.BLUE,
            'Server Room': colors.MAGENTA,
            'Break Room': colors.YELLOW
        }
    
    def _initialize_line_weights(self) -> Dict[str, float]:
        """Initialize line weights for different elements"""
        return {
            'walls': 0.5,
            'doors': 0.35,
            'windows': 0.25,
            'furniture': 0.18,
            'annotations': 0.13,
            'dimensions': 0.13,
            'area_boundaries': 0.13
        }
    
    def export_to_dxf(self, zones: List[Dict], analysis_results: Dict, 
                     output_path: str, **options) -> bool:
        """Export architectural analysis to DXF format"""
        try:
            # Create new DXF document
            doc = ezdxf.new(self.dxf_version)
            doc.units = ezdxf.units.M  # Set units to meters
            
            # Setup layers
            self._setup_layers(doc)
            
            # Get model space
            msp = doc.modelspace()
            
            # Draw zones and rooms
            self._draw_zones(msp, zones, analysis_results)
            
            # Add annotations
            self._add_annotations(msp, zones, analysis_results)
            
            # Add dimensions
            if options.get('include_dimensions', True):
                self._add_dimensions(msp, zones)
            
            # Add furniture if available
            if options.get('include_furniture', True):
                self._add_furniture_layout(msp, zones, analysis_results)
            
            # Add title block
            if options.get('include_title_block', True):
                self._add_title_block(msp, analysis_results)
            
            # Save the document
            doc.saveas(output_path)
            return True
            
        except Exception as e:
            print(f"Error exporting to DXF: {e}")
            return False
    
    def _setup_layers(self, doc):
        """Setup standard architectural layers"""
        for layer_name, properties in self.layer_standards.items():
            layer = doc.layers.add(layer_name)
            layer.color = properties['color']
            layer.linetype = properties['linetype']
            layer.lineweight = int(properties['lineweight'] * 100)  # Convert to AutoCAD units
            layer.description = properties['description']
    
    def _draw_zones(self, msp, zones: List[Dict], analysis_results: Dict):
        """Draw room zones with proper styling"""
        for i, zone in enumerate(zones):
            points = zone.get('points', [])
            if len(points) < 3:
                continue
            
            # Get room analysis
            room_info = analysis_results.get('rooms', {}).get(i, {})
            room_type = room_info.get('type', 'Unknown')
            
            # Create closed polyline for room boundary
            room_points = [(point[0], point[1], 0) for point in points]
            room_points.append(room_points[0])  # Close the polyline
            
            # Draw room boundary
            polyline = msp.add_lwpolyline(room_points)
            polyline.layer = 'A-AREA'
            
            # Fill area with hatch pattern
            hatch_color = self.color_scheme.get(room_type, colors.WHITE)
            hatch = msp.add_hatch(color=hatch_color)
            hatch.layer = 'A-AREA'
            
            # Create boundary path for hatch
            with hatch.edit_boundary() as boundary:
                boundary.add_polyline_path([(p[0], p[1]) for p in room_points[:-1]], is_closed=True)
            
            # Set hatch pattern
            hatch.set_pattern_fill("ANSI31", scale=0.1)
    
    def _add_annotations(self, msp, zones: List[Dict], analysis_results: Dict):
        """Add room labels and annotations"""
        for i, zone in enumerate(zones):
            # Calculate centroid for label placement
            points = zone.get('points', [])
            if not points:
                continue
            
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Get room information
            room_info = analysis_results.get('rooms', {}).get(i, {})
            room_type = room_info.get('type', 'Unknown')
            area = zone.get('area', 0)
            
            # Create room label
            room_number = f"R{i+1:03d}"
            room_label = f"{room_number}\\P{room_type}\\P{area:.1f} m²"
            
            # Add text annotation
            text = msp.add_mtext(
                room_label,
                insert=(centroid_x, centroid_y, 0),
                char_height=0.5,
                width=3.0
            )
            text.layer = 'A-ANNO-TEXT'
            text.dxf.attachment_point = 5  # Middle center
    
    def _add_dimensions(self, msp, zones: List[Dict]):
        """Add dimensional annotations"""
        for i, zone in enumerate(zones):
            bounds = zone.get('bounds')
            if not bounds:
                continue
            
            x_min, y_min, x_max, y_max = bounds
            width = x_max - x_min
            height = y_max - y_min
            
            # Add width dimension
            width_dim = msp.add_linear_dim(
                base=(x_min, y_min - 1.0, 0),
                p1=(x_min, y_min, 0),
                p2=(x_max, y_min, 0),
                text=f"{width:.2f}m",
                dimstyle="EZDXF"
            )
            width_dim.layer = 'A-ANNO-DIMS'
            
            # Add height dimension
            height_dim = msp.add_linear_dim(
                base=(x_min - 1.0, y_min, 0),
                p1=(x_min, y_min, 0),
                p2=(x_min, y_max, 0),
                text=f"{height:.2f}m",
                dimstyle="EZDXF"
            )
            height_dim.layer = 'A-ANNO-DIMS'
    
    def _add_furniture_layout(self, msp, zones: List[Dict], analysis_results: Dict):
        """Add furniture layout from analysis results"""
        furniture_data = analysis_results.get('furniture_optimization', {})
        
        for i, zone in enumerate(zones):
            zone_furniture = furniture_data.get(f'zone_{i}', {})
            placements = zone_furniture.get('placements', [])
            
            for placement in placements:
                # Get furniture dimensions and position
                x = placement.get('x', 0)
                y = placement.get('y', 0)
                width = placement.get('width', 1.0)
                depth = placement.get('depth', 1.0)
                rotation = placement.get('rotation', 0)
                furniture_type = placement.get('type', 'Generic')
                
                # Create furniture representation
                furniture_points = [
                    (x, y, 0),
                    (x + width, y, 0),
                    (x + width, y + depth, 0),
                    (x, y + depth, 0)
                ]
                
                furniture_rect = msp.add_lwpolyline(furniture_points, close=True)
                furniture_rect.layer = 'A-FURN'
                
                # Rotate if needed
                if rotation != 0:
                    center = (x + width/2, y + depth/2, 0)
                    furniture_rect.rotate(center, rotation)
                
                # Add furniture label
                label_x = x + width/2
                label_y = y + depth/2
                
                furniture_label = msp.add_text(
                    furniture_type[:8],  # Abbreviated name
                    insert=(label_x, label_y, 0),
                    height=0.2
                )
                furniture_label.layer = 'A-FURN'
                furniture_label.dxf.halign = 1  # Center
                furniture_label.dxf.valign = 1  # Middle
    
    def _add_title_block(self, msp, analysis_results: Dict):
        """Add professional title block"""
        # Title block dimensions and position
        tb_width = 8.0
        tb_height = 4.0
        tb_x = -2.0
        tb_y = -6.0
        
        # Draw title block border
        title_block = msp.add_lwpolyline([
            (tb_x, tb_y, 0),
            (tb_x + tb_width, tb_y, 0),
            (tb_x + tb_width, tb_y + tb_height, 0),
            (tb_x, tb_y + tb_height, 0)
        ], close=True)
        title_block.layer = 'A-ANNO-TEXT'
        
        # Add title block information
        project_name = analysis_results.get('project_name', 'DWG Analysis Project')
        drawing_title = "Architectural Space Analysis"
        drawing_number = "A-001"
        scale = "1:100"
        date = "2025-06-23"
        
        # Project name
        msp.add_text(
            project_name,
            insert=(tb_x + 0.2, tb_y + 3.5, 0),
            height=0.4
        ).layer = 'A-ANNO-TEXT'
        
        # Drawing title
        msp.add_text(
            drawing_title,
            insert=(tb_x + 0.2, tb_y + 2.8, 0),
            height=0.3
        ).layer = 'A-ANNO-TEXT'
        
        # Drawing number
        msp.add_text(
            f"Drawing No: {drawing_number}",
            insert=(tb_x + 0.2, tb_y + 2.2, 0),
            height=0.2
        ).layer = 'A-ANNO-TEXT'
        
        # Scale
        msp.add_text(
            f"Scale: {scale}",
            insert=(tb_x + 0.2, tb_y + 1.8, 0),
            height=0.2
        ).layer = 'A-ANNO-TEXT'
        
        # Date
        msp.add_text(
            f"Date: {date}",
            insert=(tb_x + 0.2, tb_y + 1.4, 0),
            height=0.2
        ).layer = 'A-ANNO-TEXT'
        
        # Software credit
        msp.add_text(
            "Generated by AI Architectural Space Analyzer Pro",
            insert=(tb_x + 0.2, tb_y + 0.2, 0),
            height=0.15
        ).layer = 'A-ANNO-TEXT'
    
    def export_to_svg(self, zones: List[Dict], analysis_results: Dict, output_path: str) -> bool:
        """Export architectural drawing to SVG format"""
        try:
            # Create DXF first
            temp_dxf = output_path.replace('.svg', '_temp.dxf')
            if not self.export_to_dxf(zones, analysis_results, temp_dxf):
                return False
            
            # Convert DXF to SVG using matplotlib backend
            doc = ezdxf.readfile(temp_dxf)
            msp = doc.modelspace()
            
            # Setup matplotlib backend
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            
            # Create render context
            ctx = RenderContext(doc)
            ctx.set_current_layout(msp)
            
            # Create matplotlib backend
            backend = MatplotlibBackend(ax)
            
            # Render the drawing
            Frontend(ctx, backend).draw_layout(msp, finalize=True)
            
            # Save as SVG
            plt.savefig(output_path, format='svg', bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            # Clean up temporary file
            os.remove(temp_dxf)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to SVG: {e}")
            return False
    
    def export_to_pdf(self, zones: List[Dict], analysis_results: Dict, output_path: str) -> bool:
        """Export architectural drawing to PDF format"""
        try:
            # Create DXF first
            temp_dxf = output_path.replace('.pdf', '_temp.dxf')
            if not self.export_to_dxf(zones, analysis_results, temp_dxf):
                return False
            
            # Convert DXF to PDF using matplotlib backend
            doc = ezdxf.readfile(temp_dxf)
            msp = doc.modelspace()
            
            # Setup matplotlib backend with high DPI for PDF
            fig = plt.figure(figsize=(11, 8.5))  # Letter size
            ax = fig.add_subplot(111)
            
            # Create render context
            ctx = RenderContext(doc)
            ctx.set_current_layout(msp)
            
            # Create matplotlib backend
            backend = MatplotlibBackend(ax)
            
            # Render the drawing
            Frontend(ctx, backend).draw_layout(msp, finalize=True)
            
            # Improve the appearance
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title('Architectural Space Analysis', fontsize=16, fontweight='bold')
            
            # Save as PDF
            plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                       facecolor='white', edgecolor='none', dpi=300)
            plt.close(fig)
            
            # Clean up temporary file
            os.remove(temp_dxf)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return False
    
    def create_drawing_package(self, zones: List[Dict], analysis_results: Dict, 
                             output_directory: str) -> Dict[str, str]:
        """Create complete drawing package with multiple formats"""
        
        # Ensure output directory exists
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Base filename
        base_name = "architectural_analysis"
        
        # Export to different formats
        formats = {
            'dxf': f"{base_name}.dxf",
            'svg': f"{base_name}.svg", 
            'pdf': f"{base_name}.pdf"
        }
        
        exported_files = {}
        
        for format_name, filename in formats.items():
            output_path = os.path.join(output_directory, filename)
            
            if format_name == 'dxf':
                success = self.export_to_dxf(zones, analysis_results, output_path,
                                           include_dimensions=True,
                                           include_furniture=True,
                                           include_title_block=True)
            elif format_name == 'svg':
                success = self.export_to_svg(zones, analysis_results, output_path)
            elif format_name == 'pdf':
                success = self.export_to_pdf(zones, analysis_results, output_path)
            
            if success:
                exported_files[format_name] = output_path
            else:
                print(f"Failed to export {format_name} format")
        
        # Create summary JSON
        summary = {
            'exported_files': exported_files,
            'analysis_summary': {
                'total_zones': len(zones),
                'total_area': sum(zone.get('area', 0) for zone in zones),
                'room_types': list(set(
                    analysis_results.get('rooms', {}).get(i, {}).get('type', 'Unknown')
                    for i in range(len(zones))
                ))
            },
            'export_timestamp': str(plt.datetime.datetime.now()),
            'software': 'AI Architectural Space Analyzer Pro'
        }
        
        summary_path = os.path.join(output_directory, 'export_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        exported_files['summary'] = summary_path
        
        return exported_files