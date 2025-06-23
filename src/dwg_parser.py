import ezdxf
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

class DWGParser:
    """
    Parser for DWG and DXF files using ezdxf library
    """
    
    def __init__(self):
        self.supported_entities = [
            'LWPOLYLINE', 'POLYLINE', 'LINE', 'ARC', 'CIRCLE', 
            'ELLIPSE', 'SPLINE', 'HATCH'
        ]
    
    def parse_file(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parse DWG/DXF file and extract zones (closed polygons)
        
        Args:
            file_bytes: Raw file content as bytes
            filename: Original filename
            
        Returns:
            List of zone dictionaries with points and metadata
        """
        zones = []
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        try:
            if filename.lower().endswith('.dwg'):
                raise Exception("DWG files are not directly supported. Please convert to DXF format first.")
            
            # Try to read the DXF file
            try:
                doc = ezdxf.readfile(temp_file_path)
            except ezdxf.DXFStructureError:
                # Try with recovery mode for corrupted files
                doc = ezdxf.recover.readfile(temp_file_path)
            
            modelspace = doc.modelspace()
            
            # Extract layers information
            layers = self._extract_layers(doc)
            
            # Parse different entity types
            zones.extend(self._parse_lwpolylines(modelspace))
            zones.extend(self._parse_polylines(modelspace))
            zones.extend(self._parse_hatches(modelspace))
            zones.extend(self._parse_closed_shapes(modelspace))
            
            # Add layer information to zones
            for zone in zones:
                layer_name = zone.get('layer', '0')
                if layer_name in layers:
                    zone['layer_info'] = layers[layer_name]
            
        except ezdxf.DXFError as e:
            raise Exception(f"DXF parsing error: {str(e)}")
        except Exception as e:
            raise Exception(f"File parsing error: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return self._validate_and_clean_zones(zones)
    
    def parse_file_from_path(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse DWG/DXF file from file path"""
        zones = []
        
        try:
            # Try to read the DXF/DWG file
            try:
                doc = ezdxf.readfile(file_path)
            except ezdxf.DXFStructureError:
                # Try with recovery mode for corrupted files
                doc = ezdxf.recover.readfile(file_path)
            
            modelspace = doc.modelspace()
            
            # Extract layers information
            layers = self._extract_layers(doc)
            
            # Parse different entity types
            zones.extend(self._parse_lwpolylines(modelspace))
            zones.extend(self._parse_polylines(modelspace))
            zones.extend(self._parse_hatches(modelspace))
            zones.extend(self._parse_closed_shapes(modelspace))
            
            # Filter out very small zones (likely noise)
            zones = [zone for zone in zones if zone.get('area', 0) > 0.1]
            
            # Add zone IDs
            for i, zone in enumerate(zones):
                zone['id'] = f"zone_{i+1}"
                zone['layers'] = layers
                
        except Exception as e:
            print(f"Error parsing DWG file: {str(e)}")
            return []
        
        return self._validate_and_clean_zones(zones)
    
    def _extract_layers(self, doc) -> Dict[str, Dict]:
        """Extract layer information from the document"""
        layers = {}
        
        try:
            for layer in doc.layers:
                layers[layer.dxf.name] = {
                    'name': layer.dxf.name,
                    'color': getattr(layer.dxf, 'color', 7),
                    'linetype': getattr(layer.dxf, 'linetype', 'CONTINUOUS'),
                    'visible': not getattr(layer.dxf, 'flags', 0) & 1  # Check if frozen
                }
        except:
            # Fallback if layer extraction fails
            layers['0'] = {'name': '0', 'color': 7, 'linetype': 'CONTINUOUS', 'visible': True}
        
        return layers
    
    def _parse_lwpolylines(self, modelspace) -> List[Dict[str, Any]]:
        """Parse LWPOLYLINE entities"""
        zones = []
        
        for entity in modelspace.query('LWPOLYLINE'):
            if entity.closed:
                try:
                    points = [(point[0], point[1]) for point in entity]
                    if len(points) >= 3:  # Minimum for a polygon
                        zones.append({
                            'points': points,
                            'layer': entity.dxf.layer,
                            'entity_type': 'LWPOLYLINE',
                            'closed': True,
                            'area': self._calculate_polygon_area(points)
                        })
                except Exception as e:
                    continue  # Skip problematic entities
        
        return zones
    
    def _parse_polylines(self, modelspace) -> List[Dict[str, Any]]:
        """Parse POLYLINE entities"""
        zones = []
        
        for entity in modelspace.query('POLYLINE'):
            if entity.is_closed:
                try:
                    points = [(vertex.dxf.location[0], vertex.dxf.location[1]) 
                             for vertex in entity.vertices]
                    if len(points) >= 3:
                        zones.append({
                            'points': points,
                            'layer': entity.dxf.layer,
                            'entity_type': 'POLYLINE',
                            'closed': True,
                            'area': self._calculate_polygon_area(points)
                        })
                except Exception as e:
                    continue
        
        return zones
    
    def _parse_hatches(self, modelspace) -> List[Dict[str, Any]]:
        """Parse HATCH entities to extract boundary polygons"""
        zones = []
        
        for entity in modelspace.query('HATCH'):
            try:
                # Get boundary paths
                for boundary_path in entity.paths:
                    if boundary_path.path_type_flags & 2:  # External boundary
                        points = []
                        for edge in boundary_path.edges:
                            if edge.EDGE_TYPE == 'LineEdge':
                                points.append((edge.start[0], edge.start[1]))
                            elif edge.EDGE_TYPE == 'ArcEdge':
                                # Approximate arc with line segments
                                arc_points = self._approximate_arc(edge)
                                points.extend(arc_points)
                        
                        if len(points) >= 3:
                            zones.append({
                                'points': points,
                                'layer': entity.dxf.layer,
                                'entity_type': 'HATCH',
                                'closed': True,
                                'area': self._calculate_polygon_area(points)
                            })
            except Exception as e:
                continue
        
        return zones
    
    def _parse_closed_shapes(self, modelspace) -> List[Dict[str, Any]]:
        """Parse other closed shapes like circles and rectangles"""
        zones = []
        
        # Parse circles
        for entity in modelspace.query('CIRCLE'):
            try:
                center = entity.dxf.center
                radius = entity.dxf.radius
                # Approximate circle with polygon
                points = self._circle_to_polygon(center[0], center[1], radius)
                zones.append({
                    'points': points,
                    'layer': entity.dxf.layer,
                    'entity_type': 'CIRCLE',
                    'closed': True,
                    'area': 3.14159 * radius * radius
                })
            except Exception as e:
                continue
        
        # Parse ellipses
        for entity in modelspace.query('ELLIPSE'):
            try:
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                # Approximate ellipse with polygon
                points = self._ellipse_to_polygon(center, major_axis, ratio)
                zones.append({
                    'points': points,
                    'layer': entity.dxf.layer,
                    'entity_type': 'ELLIPSE',
                    'closed': True,
                    'area': self._calculate_polygon_area(points)
                })
            except Exception as e:
                continue
        
        return zones
    
    def _calculate_polygon_area(self, points: List[tuple]) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def _circle_to_polygon(self, cx: float, cy: float, radius: float, num_points: int = 32) -> List[tuple]:
        """Convert circle to polygon approximation"""
        import math
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        return points
    
    def _ellipse_to_polygon(self, center, major_axis, ratio: float, num_points: int = 32) -> List[tuple]:
        """Convert ellipse to polygon approximation"""
        import math
        points = []
        cx, cy = center[0], center[1]
        major_length = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
        minor_length = major_length * ratio
        
        # Calculate rotation angle
        angle_offset = math.atan2(major_axis[1], major_axis[0])
        
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            # Ellipse in local coordinates
            local_x = major_length * math.cos(t)
            local_y = minor_length * math.sin(t)
            
            # Rotate and translate
            x = cx + local_x * math.cos(angle_offset) - local_y * math.sin(angle_offset)
            y = cy + local_x * math.sin(angle_offset) + local_y * math.cos(angle_offset)
            points.append((x, y))
        
        return points
    
    def _approximate_arc(self, edge, num_segments: int = 8) -> List[tuple]:
        """Approximate arc edge with line segments"""
        import math
        points = []
        
        center = edge.center
        radius = edge.radius
        start_angle = edge.start_angle
        end_angle = edge.end_angle
        
        # Handle angle wrapping
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        
        angle_step = (end_angle - start_angle) / num_segments
        
        for i in range(num_segments + 1):
            angle = start_angle + i * angle_step
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        
        return points
    
    def _validate_and_clean_zones(self, zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean extracted zones"""
        valid_zones = []
        
        for zone in zones:
            # Check if zone has valid points
            if not zone.get('points') or len(zone['points']) < 3:
                continue
            
            # Remove duplicate consecutive points
            cleaned_points = []
            prev_point = None
            
            for point in zone['points']:
                if prev_point is None or (abs(point[0] - prev_point[0]) > 1e-6 or 
                                        abs(point[1] - prev_point[1]) > 1e-6):
                    cleaned_points.append(point)
                    prev_point = point
            
            if len(cleaned_points) >= 3:
                zone['points'] = cleaned_points
                zone['area'] = self._calculate_polygon_area(cleaned_points)
                
                # Add bounding box
                xs = [p[0] for p in cleaned_points]
                ys = [p[1] for p in cleaned_points]
                zone['bounds'] = (min(xs), min(ys), max(xs), max(ys))
                
                valid_zones.append(zone)
        
        return valid_zones
