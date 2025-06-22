import ezdxf
from ezdxf import units
from ezdxf.addons import r12writer
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import math
from shapely.geometry import Polygon, Point
import tempfile
import os

class CADExporter:
    """
    Advanced CAD export functionality supporting multiple formats
    including DXF, DWG, and specialized architectural formats
    """
    
    def __init__(self):
        self.supported_formats = ['DXF', 'DWG', 'SVG', 'PDF']
        self.scale_factors = {
            'mm': 1.0,
            'm': 1000.0,
            'cm': 100.0,
            'in': 25.4,
            'ft': 304.8
        }
        
        # Layer definitions for different elements
        self.layer_definitions = {
            'ZONES': {'color': 1, 'linetype': 'CONTINUOUS', 'lineweight': 0.25},
            'FURNITURE': {'color': 3, 'linetype': 'CONTINUOUS', 'lineweight': 0.18},
            'DIMENSIONS': {'color': 2, 'linetype': 'CONTINUOUS', 'lineweight': 0.13},
            'TEXT': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 0.09},
            'GRID': {'color': 8, 'linetype': 'DASHED', 'lineweight': 0.09},
            'CENTERLINES': {'color': 5, 'linetype': 'CENTER', 'lineweight': 0.13},
            'HIDDEN': {'color': 6, 'linetype': 'HIDDEN', 'lineweight': 0.13},
            'CONSTRUCTION': {'color': 9, 'linetype': 'PHANTOM', 'lineweight': 0.09}
        }
        
        # Text styles
        self.text_styles = {
            'STANDARD': {'font': 'Arial', 'height': 2.5},
            'TITLE': {'font': 'Arial', 'height': 5.0},
            'NOTES': {'font': 'Arial', 'height': 2.0},
            'DIMENSIONS': {'font': 'Arial', 'height': 1.8}
        }
    
    def export_to_dxf(self, zones: List[Dict], analysis_results: Dict, 
                     output_path: str, scale: str = 'm',
                     include_furniture: bool = True,
                     include_dimensions: bool = True,
                     include_annotations: bool = True) -> str:
        """Export architectural plan to DXF format"""
        
        # Create new DXF document
        doc = ezdxf.new('R2010')
        doc.units = units.M if scale == 'm' else units.MM
        
        # Setup layers
        self._setup_layers(doc)
        
        # Setup text styles
        self._setup_text_styles(doc)
        
        # Get model space
        msp = doc.modelspace()
        
        # Add zones
        self._add_zones_to_dxf(msp, zones, analysis_results, scale)
        
        # Add furniture if requested
        if include_furniture and analysis_results.get('placements'):
            self._add_furniture_to_dxf(msp, analysis_results['placements'], scale)
        
        # Add dimensions if requested
        if include_dimensions:
            self._add_dimensions_to_dxf(msp, zones, scale)
        
        # Add annotations if requested
        if include_annotations:
            self._add_annotations_to_dxf(msp, zones, analysis_results, scale)
        
        # Add title block
        self._add_title_block(msp, analysis_results, scale)
        
        # Add reference grid
        self._add_reference_grid(msp, zones, scale)
        
        # Save the DXF file
        doc.saveas(output_path)
        
        return output_path
    
    def _setup_layers(self, doc):
        """Setup standard architectural layers"""
        for layer_name, properties in self.layer_definitions.items():
            layer = doc.layers.new(layer_name)
            layer.color = properties['color']
            layer.linetype = properties['linetype']
            layer.lineweight = int(properties['lineweight'] * 100)  # Convert to lineweight units
    
    def _setup_text_styles(self, doc):
        """Setup text styles"""
        for style_name, properties in self.text_styles.items():
            doc.styles.new(style_name, dxfattribs={
                'font': properties['font'],
                'height': properties['height']
            })
    
    def _add_zones_to_dxf(self, msp, zones: List[Dict], analysis_results: Dict, scale: str):
        """Add zone boundaries to DXF"""
        scale_factor = self.scale_factors[scale]
        rooms = analysis_results.get('rooms', {})
        
        for i, zone in enumerate(zones):
            if not zone.get('points'):
                continue
            
            zone_name = f"Zone_{i}"
            room_info = rooms.get(zone_name, {})
            
            # Scale points
            scaled_points = [(p[0] * scale_factor, p[1] * scale_factor) for p in zone['points']]
            
            # Create polyline for zone boundary
            polyline = msp.add_lwpolyline(scaled_points + [scaled_points[0]])
            polyline.dxf.layer = 'ZONES'
            polyline.closed = True
            
            # Add zone label
            poly = Polygon(zone['points'])
            centroid = poly.centroid
            centroid_scaled = (centroid.x * scale_factor, centroid.y * scale_factor)
            
            # Room type label
            room_type = room_info.get('type', 'Unknown')
            area = room_info.get('area', 0)
            
            label_text = f"{room_type}\n{area:.1f} {scale}²"
            
            text = msp.add_text(
                label_text,
                dxfattribs={
                    'layer': 'TEXT',
                    'style': 'STANDARD',
                    'height': 2.5 * scale_factor / 1000,  # Adjust for scale
                    'halign': 1,  # Center
                    'valign': 2   # Middle
                }
            )
            text.set_pos(centroid_scaled)
    
    def _add_furniture_to_dxf(self, msp, placements: Dict[str, List[Dict]], scale: str):
        """Add furniture placements to DXF"""
        scale_factor = self.scale_factors[scale]
        
        for zone_name, zone_placements in placements.items():
            for placement in zone_placements:
                # Scale furniture coordinates
                scaled_coords = [
                    (p[0] * scale_factor, p[1] * scale_factor) 
                    for p in placement['box_coords']
                ]
                
                # Create polyline for furniture
                furniture = msp.add_lwpolyline(scaled_coords + [scaled_coords[0]])
                furniture.dxf.layer = 'FURNITURE'
                furniture.closed = True
                
                # Add hatch pattern for furniture
                hatch = msp.add_hatch(color=3)
                hatch.dxf.layer = 'FURNITURE'
                hatch.paths.add_polyline_path(scaled_coords + [scaled_coords[0]])
                hatch.set_pattern_fill('ANSI31', scale=0.5)
                
                # Add furniture centerlines
                center_x = sum(p[0] for p in scaled_coords) / len(scaled_coords)
                center_y = sum(p[1] for p in scaled_coords) / len(scaled_coords)
                
                # Cross centerlines
                size = placement['size']
                half_length = size[0] * scale_factor / 2
                half_width = size[1] * scale_factor / 2
                
                # Horizontal centerline
                msp.add_line(
                    (center_x - half_length, center_y),
                    (center_x + half_length, center_y),
                    dxfattribs={'layer': 'CENTERLINES'}
                )
                
                # Vertical centerline
                msp.add_line(
                    (center_x, center_y - half_width),
                    (center_x, center_y + half_width),
                    dxfattribs={'layer': 'CENTERLINES'}
                )
    
    def _add_dimensions_to_dxf(self, msp, zones: List[Dict], scale: str):
        """Add dimensions to zone boundaries"""
        scale_factor = self.scale_factors[scale]
        
        for zone in zones:
            if not zone.get('points') or len(zone['points']) < 3:
                continue
            
            points = zone['points']
            
            # Add dimensions for major edges
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                # Calculate edge length
                length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                # Only dimension significant edges (> 1m)
                if length > 1.0:
                    # Scale points
                    p1_scaled = (p1[0] * scale_factor, p1[1] * scale_factor)
                    p2_scaled = (p2[0] * scale_factor, p2[1] * scale_factor)
                    
                    # Calculate dimension line offset
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    # Perpendicular offset for dimension line
                    offset_distance = 1.0 * scale_factor  # 1m offset
                    if abs(dx) > abs(dy):  # Horizontal edge
                        offset = (0, offset_distance if dy >= 0 else -offset_distance)
                    else:  # Vertical edge
                        offset = (offset_distance if dx >= 0 else -offset_distance, 0)
                    
                    dim_p1 = (p1_scaled[0] + offset[0], p1_scaled[1] + offset[1])
                    dim_p2 = (p2_scaled[0] + offset[0], p2_scaled[1] + offset[1])
                    
                    # Add dimension
                    dim = msp.add_linear_dim(
                        base=dim_p1,
                        p1=p1_scaled,
                        p2=p2_scaled,
                        dxfattribs={'layer': 'DIMENSIONS'}
                    )
                    
                    # Format dimension text
                    dim.render()
    
    def _add_annotations_to_dxf(self, msp, zones: List[Dict], analysis_results: Dict, scale: str):
        """Add annotations and notes"""
        scale_factor = self.scale_factors[scale]
        
        # Add analysis summary
        if analysis_results.get('total_boxes'):
            summary_text = (
                f"ANALYSIS SUMMARY\n"
                f"Total Boxes: {analysis_results['total_boxes']}\n"
                f"Total Zones: {len(zones)}\n"
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            # Position in upper right corner
            bounds = self._calculate_overall_bounds(zones)
            if bounds:
                max_x, max_y = bounds[2] * scale_factor, bounds[3] * scale_factor
                
                msp.add_text(
                    summary_text,
                    dxfattribs={
                        'layer': 'TEXT',
                        'style': 'NOTES',
                        'height': 2.0 * scale_factor / 1000
                    }
                ).set_pos((max_x + 5 * scale_factor, max_y))
        
        # Add room-specific annotations
        rooms = analysis_results.get('rooms', {})
        placements = analysis_results.get('placements', {})
        
        for i, zone in enumerate(zones):
            zone_name = f"Zone_{i}"
            zone_placements = placements.get(zone_name, [])
            room_info = rooms.get(zone_name, {})
            
            if zone_placements and room_info:
                # Add placement count annotation
                poly = Polygon(zone['points'])
                centroid = poly.centroid
                
                # Offset annotation below room label
                annotation_pos = (
                    centroid.x * scale_factor,
                    (centroid.y - 2) * scale_factor
                )
                
                annotation_text = f"Furniture: {len(zone_placements)} items"
                
                msp.add_text(
                    annotation_text,
                    dxfattribs={
                        'layer': 'TEXT',
                        'style': 'NOTES',
                        'height': 1.5 * scale_factor / 1000,
                        'halign': 1,
                        'valign': 2
                    }
                ).set_pos(annotation_pos)
    
    def _add_title_block(self, msp, analysis_results: Dict, scale: str):
        """Add standard architectural title block"""
        scale_factor = self.scale_factors[scale]
        
        # Title block dimensions (scaled)
        tb_width = 200 * scale_factor / 1000
        tb_height = 50 * scale_factor / 1000
        
        # Position at bottom right
        bounds = self._calculate_overall_bounds([])
        if not bounds:
            # Default position
            tb_x, tb_y = 0, -tb_height - 10 * scale_factor / 1000
        else:
            tb_x = bounds[2] * scale_factor - tb_width
            tb_y = bounds[1] * scale_factor - tb_height - 10 * scale_factor / 1000
        
        # Draw title block border
        tb_points = [
            (tb_x, tb_y),
            (tb_x + tb_width, tb_y),
            (tb_x + tb_width, tb_y + tb_height),
            (tb_x, tb_y + tb_height),
            (tb_x, tb_y)
        ]
        
        msp.add_lwpolyline(tb_points, dxfattribs={'layer': 'TEXT'})
        
        # Add title block content
        title_text = "AI ARCHITECTURAL SPACE ANALYZER"
        subtitle_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Project title
        msp.add_text(
            title_text,
            dxfattribs={
                'layer': 'TEXT',
                'style': 'TITLE',
                'height': 4.0 * scale_factor / 1000,
                'halign': 1
            }
        ).set_pos((tb_x + tb_width/2, tb_y + tb_height*0.7))
        
        # Subtitle
        msp.add_text(
            subtitle_text,
            dxfattribs={
                'layer': 'TEXT',
                'style': 'STANDARD',
                'height': 2.0 * scale_factor / 1000,
                'halign': 1
            }
        ).set_pos((tb_x + tb_width/2, tb_y + tb_height*0.4))
        
        # Scale indicator
        scale_text = f"SCALE: 1:{int(scale_factor)}"
        msp.add_text(
            scale_text,
            dxfattribs={
                'layer': 'TEXT',
                'style': 'STANDARD',
                'height': 1.8 * scale_factor / 1000,
                'halign': 1
            }
        ).set_pos((tb_x + tb_width/2, tb_y + tb_height*0.1))
    
    def _add_reference_grid(self, msp, zones: List[Dict], scale: str):
        """Add reference grid"""
        scale_factor = self.scale_factors[scale]
        
        bounds = self._calculate_overall_bounds(zones)
        if not bounds:
            return
        
        min_x, min_y, max_x, max_y = bounds
        
        # Scale bounds
        min_x *= scale_factor
        min_y *= scale_factor
        max_x *= scale_factor
        max_y *= scale_factor
        
        # Grid spacing (5m in real units)
        grid_spacing = 5.0 * scale_factor
        
        # Extend grid beyond bounds
        grid_min_x = math.floor(min_x / grid_spacing) * grid_spacing
        grid_max_x = math.ceil(max_x / grid_spacing) * grid_spacing
        grid_min_y = math.floor(min_y / grid_spacing) * grid_spacing
        grid_max_y = math.ceil(max_y / grid_spacing) * grid_spacing
        
        # Add vertical grid lines
        x = grid_min_x
        while x <= grid_max_x:
            msp.add_line(
                (x, grid_min_y - grid_spacing),
                (x, grid_max_y + grid_spacing),
                dxfattribs={'layer': 'GRID'}
            )
            x += grid_spacing
        
        # Add horizontal grid lines
        y = grid_min_y
        while y <= grid_max_y:
            msp.add_line(
                (grid_min_x - grid_spacing, y),
                (grid_max_x + grid_spacing, y),
                dxfattribs={'layer': 'GRID'}
            )
            y += grid_spacing
        
        # Add grid labels
        # Vertical labels
        x = grid_min_x
        label_num = 1
        while x <= grid_max_x:
            msp.add_text(
                str(label_num),
                dxfattribs={
                    'layer': 'TEXT',
                    'style': 'NOTES',
                    'height': 1.5 * scale_factor / 1000,
                    'halign': 1,
                    'valign': 2
                }
            ).set_pos((x, grid_min_y - grid_spacing/2))
            x += grid_spacing
            label_num += 1
        
        # Horizontal labels
        y = grid_min_y
        label_char = ord('A')
        while y <= grid_max_y:
            msp.add_text(
                chr(label_char),
                dxfattribs={
                    'layer': 'TEXT',
                    'style': 'NOTES',
                    'height': 1.5 * scale_factor / 1000,
                    'halign': 1,
                    'valign': 2
                }
            ).set_pos((grid_min_x - grid_spacing/2, y))
            y += grid_spacing
            label_char += 1
    
    def _calculate_overall_bounds(self, zones: List[Dict]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate overall bounds of all zones"""
        if not zones:
            return None
        
        all_points = []
        for zone in zones:
            if zone.get('points'):
                all_points.extend(zone['points'])
        
        if not all_points:
            return None
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def export_to_svg(self, zones: List[Dict], analysis_results: Dict, 
                     output_path: str, width: int = 800, height: int = 600) -> str:
        """Export to SVG format for web display"""
        
        # Calculate bounds and scale
        bounds = self._calculate_overall_bounds(zones)
        if not bounds:
            return ""
        
        min_x, min_y, max_x, max_y = bounds
        plan_width = max_x - min_x
        plan_height = max_y - min_y
        
        # Calculate scale to fit in desired dimensions
        scale_x = width * 0.8 / plan_width if plan_width > 0 else 1
        scale_y = height * 0.8 / plan_height if plan_height > 0 else 1
        scale = min(scale_x, scale_y)
        
        # SVG header
        svg_content = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<defs>',
            f'  <style>',
            f'    .zone {{ fill: none; stroke: #000; stroke-width: 1; }}',
            f'    .furniture {{ fill: #90EE90; stroke: #006400; stroke-width: 0.5; }}',
            f'    .text {{ font-family: Arial; font-size: 12px; fill: #000; }}',
            f'    .grid {{ stroke: #ccc; stroke-width: 0.5; stroke-dasharray: 2,2; }}',
            f'  </style>',
            f'</defs>',
            f'<g transform="translate({width*0.1}, {height*0.1}) scale({scale}, {-scale}) translate({-min_x}, {-max_y})">'
        ]
        
        # Add zones
        rooms = analysis_results.get('rooms', {})
        for i, zone in enumerate(zones):
            if not zone.get('points'):
                continue
            
            points_str = ' '.join([f"{p[0]},{p[1]}" for p in zone['points']])
            svg_content.append(f'  <polygon points="{points_str}" class="zone"/>')
            
            # Add zone label
            zone_name = f"Zone_{i}"
            room_info = rooms.get(zone_name, {})
            poly = Polygon(zone['points'])
            centroid = poly.centroid
            
            room_type = room_info.get('type', 'Unknown')
            svg_content.append(
                f'  <text x="{centroid.x}" y="{centroid.y}" class="text" '
                f'text-anchor="middle" transform="scale(1,-1)">{room_type}</text>'
            )
        
        # Add furniture
        if analysis_results.get('placements'):
            for zone_placements in analysis_results['placements'].values():
                for placement in zone_placements:
                    points_str = ' '.join([f"{p[0]},{p[1]}" for p in placement['box_coords']])
                    svg_content.append(f'  <polygon points="{points_str}" class="furniture"/>')
        
        svg_content.extend([
            '</g>',
            '</svg>'
        ])
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(svg_content))
        
        return output_path
    
    def export_3d_model(self, zones: List[Dict], analysis_results: Dict,
                       output_path: str, format: str = 'OBJ') -> str:
        """Export 3D model in OBJ or STL format"""
        
        if format.upper() not in ['OBJ', 'STL']:
            raise ValueError("Supported 3D formats: OBJ, STL")
        
        vertices = []
        faces = []
        vertex_count = 0
        
        # Default heights
        floor_height = 0.1  # 10cm floor thickness
        wall_height = 3.0   # 3m wall height
        furniture_height = 0.75  # 75cm furniture height
        
        # Add floor and walls for each zone
        for zone in zones:
            if not zone.get('points'):
                continue
            
            points = zone['points']
            
            # Floor vertices (bottom)
            floor_vertices_bottom = []
            for point in points:
                vertices.append((point[0], point[1], 0))
                floor_vertices_bottom.append(vertex_count)
                vertex_count += 1
            
            # Floor vertices (top)
            floor_vertices_top = []
            for point in points:
                vertices.append((point[0], point[1], floor_height))
                floor_vertices_top.append(vertex_count)
                vertex_count += 1
            
            # Create floor faces
            for i in range(len(points) - 1):
                # Bottom face (reversed for correct normal)
                faces.append([floor_vertices_bottom[i+1], floor_vertices_bottom[i], floor_vertices_bottom[0]])
                
                # Top face
                faces.append([floor_vertices_top[i], floor_vertices_top[i+1], floor_vertices_top[0]])
                
                # Side faces
                faces.append([
                    floor_vertices_bottom[i], floor_vertices_bottom[i+1],
                    floor_vertices_top[i+1], floor_vertices_top[i]
                ])
            
            # Wall vertices
            wall_vertices_bottom = floor_vertices_top[:]
            wall_vertices_top = []
            for point in points:
                vertices.append((point[0], point[1], wall_height))
                wall_vertices_top.append(vertex_count)
                vertex_count += 1
            
            # Create wall faces (exterior walls only)
            for i in range(len(points) - 1):
                faces.append([
                    wall_vertices_bottom[i], wall_vertices_bottom[i+1],
                    wall_vertices_top[i+1], wall_vertices_top[i]
                ])
        
        # Add furniture as 3D boxes
        if analysis_results.get('placements'):
            for zone_placements in analysis_results['placements'].values():
                for placement in zone_placements:
                    coords = placement['box_coords']
                    
                    # Create 3D box for furniture
                    box_vertices = []
                    
                    # Bottom vertices
                    for coord in coords:
                        vertices.append((coord[0], coord[1], floor_height))
                        box_vertices.append(vertex_count)
                        vertex_count += 1
                    
                    # Top vertices
                    for coord in coords:
                        vertices.append((coord[0], coord[1], floor_height + furniture_height))
                        box_vertices.append(vertex_count)
                        vertex_count += 1
                    
                    # Create box faces
                    # Bottom face
                    faces.append([box_vertices[0], box_vertices[1], box_vertices[2], box_vertices[3]])
                    
                    # Top face
                    faces.append([box_vertices[4], box_vertices[7], box_vertices[6], box_vertices[5]])
                    
                    # Side faces
                    faces.append([box_vertices[0], box_vertices[4], box_vertices[5], box_vertices[1]])
                    faces.append([box_vertices[1], box_vertices[5], box_vertices[6], box_vertices[2]])
                    faces.append([box_vertices[2], box_vertices[6], box_vertices[7], box_vertices[3]])
                    faces.append([box_vertices[3], box_vertices[7], box_vertices[4], box_vertices[0]])
        
        # Write OBJ file
        if format.upper() == 'OBJ':
            with open(output_path, 'w') as f:
                f.write("# AI Architectural Space Analyzer - 3D Export\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                
                f.write("\n")
                
                # Write faces (OBJ uses 1-based indexing)
                for face in faces:
                    if len(face) == 3:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                    elif len(face) == 4:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")
        
        return output_path
    
    def create_technical_drawing_package(self, zones: List[Dict], analysis_results: Dict,
                                       output_dir: str) -> Dict[str, str]:
        """Create complete technical drawing package"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        package_files = {}
        
        # Floor plan (DXF)
        floor_plan_path = os.path.join(output_dir, "floor_plan.dxf")
        self.export_to_dxf(zones, analysis_results, floor_plan_path, 
                          include_furniture=True, include_dimensions=True)
        package_files['floor_plan_dxf'] = floor_plan_path
        
        # Furniture plan (DXF)
        furniture_plan_path = os.path.join(output_dir, "furniture_plan.dxf")
        self.export_to_dxf(zones, analysis_results, furniture_plan_path,
                          include_furniture=True, include_dimensions=False)
        package_files['furniture_plan_dxf'] = furniture_plan_path
        
        # Dimensions plan (DXF)
        dimensions_plan_path = os.path.join(output_dir, "dimensions_plan.dxf")
        self.export_to_dxf(zones, analysis_results, dimensions_plan_path,
                          include_furniture=False, include_dimensions=True)
        package_files['dimensions_plan_dxf'] = dimensions_plan_path
        
        # Web preview (SVG)
        svg_preview_path = os.path.join(output_dir, "preview.svg")
        self.export_to_svg(zones, analysis_results, svg_preview_path)
        package_files['svg_preview'] = svg_preview_path
        
        # 3D model (OBJ)
        model_3d_path = os.path.join(output_dir, "model_3d.obj")
        self.export_3d_model(zones, analysis_results, model_3d_path)
        package_files['3d_model_obj'] = model_3d_path
        
        # Analysis report (JSON)
        report_path = os.path.join(output_dir, "analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'export_date': datetime.now().isoformat(),
                'zones_count': len(zones),
                'analysis_results': analysis_results,
                'package_contents': list(package_files.keys())
            }, f, indent=2)
        package_files['analysis_report'] = report_path
        
        return package_files