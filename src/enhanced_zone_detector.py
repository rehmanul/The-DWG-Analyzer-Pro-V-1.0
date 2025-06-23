"""
Enhanced Zone Detection for Architectural Plans
Improves accuracy of room/space detection from DWG/DXF files
"""

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, polygonize
from typing import List, Dict, Any, Tuple
import networkx as nx


class EnhancedZoneDetector:
    """Enhanced zone detection with improved accuracy for architectural plans"""
    
    def __init__(self):
        self.min_area = 1.0  # Minimum area for a valid room (square meters)
        self.max_area = 10000.0  # Maximum area for a valid room
        self.wall_thickness_tolerance = 0.1  # Tolerance for wall thickness detection
        
    def detect_zones_from_entities(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Enhanced zone detection from CAD entities
        
        Args:
            entities: List of CAD entities (lines, polylines, etc.)
            
        Returns:
            List of detected zones with enhanced metadata
        """
        zones = []
        
        # Method 1: Direct closed polygon detection
        closed_polygons = self._find_closed_polygons(entities)
        zones.extend(closed_polygons)
        
        # Method 2: Line network analysis for room boundaries
        if len(zones) < 5:  # If few zones found, try line analysis
            line_based_zones = self._detect_zones_from_lines(entities)
            zones.extend(line_based_zones)
        
        # Method 3: Hatch-based room detection
        hatch_zones = self._detect_zones_from_hatches(entities)
        zones.extend(hatch_zones)
        
        # Method 4: Text-guided zone detection
        text_zones = self._detect_zones_near_text(entities)
        zones.extend(text_zones)
        
        # Remove duplicates and validate
        zones = self._remove_duplicate_zones(zones)
        zones = self._validate_zones(zones)
        zones = self._classify_zones(zones)
        
        return zones
    
    def _find_closed_polygons(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Find directly closed polygons from entities"""
        zones = []
        
        for entity in entities:
            if entity.get('entity_type') in ['LWPOLYLINE', 'POLYLINE']:
                if entity.get('closed', False) or self._is_polygon_closed(entity.get('points', [])):
                    points = entity.get('points', [])
                    if len(points) >= 3:
                        zone = self._create_zone_from_points(points, entity)
                        if zone:
                            zones.append(zone)
        
        return zones
    
    def _detect_zones_from_lines(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Detect zones by analyzing line networks to find enclosed areas"""
        zones = []
        
        # Extract all line segments
        lines = []
        for entity in entities:
            if entity.get('entity_type') == 'LINE':
                start = entity.get('start_point')
                end = entity.get('end_point')
                if start and end:
                    lines.append(LineString([start, end]))
            elif entity.get('entity_type') in ['LWPOLYLINE', 'POLYLINE']:
                points = entity.get('points', [])
                for i in range(len(points) - 1):
                    lines.append(LineString([points[i], points[i + 1]]))
        
        if not lines:
            return zones
        
        # Create network graph of connected line segments
        line_network = self._create_line_network(lines)
        
        # Find cycles in the network (potential rooms)
        cycles = self._find_cycles_in_network(line_network)
        
        # Convert cycles to zones
        for cycle in cycles:
            if len(cycle) >= 3:
                zone = self._create_zone_from_cycle(cycle, line_network)
                if zone:
                    zones.append(zone)
        
        return zones
    
    def _detect_zones_from_hatches(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Detect zones from hatch patterns (often used for room fills)"""
        zones = []
        
        for entity in entities:
            if entity.get('entity_type') == 'HATCH':
                boundary_paths = entity.get('boundary_paths', [])
                for path in boundary_paths:
                    if len(path) >= 3:
                        zone = self._create_zone_from_points(path, entity)
                        if zone:
                            zone['zone_type'] = 'hatch_based'
                            zones.append(zone)
        
        return zones
    
    def _detect_zones_near_text(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Detect zones near text labels (room names/numbers)"""
        zones = []
        
        # Find text entities
        text_entities = [e for e in entities if e.get('entity_type') == 'TEXT']
        
        for text_entity in text_entities:
            text_point = text_entity.get('insertion_point')
            text_content = text_entity.get('text', '').strip()
            
            if text_point and self._is_likely_room_label(text_content):
                # Find enclosing geometry around this text
                enclosing_zone = self._find_enclosing_geometry(text_point, entities)
                if enclosing_zone:
                    enclosing_zone['room_label'] = text_content
                    enclosing_zone['zone_type'] = 'text_guided'
                    zones.append(enclosing_zone)
        
        return zones
    
    def _create_line_network(self, lines: List[LineString]) -> nx.Graph:
        """Create a network graph from line segments"""
        G = nx.Graph()
        
        # Add nodes for line endpoints
        for i, line in enumerate(lines):
            start = tuple(line.coords[0])
            end = tuple(line.coords[-1])
            
            G.add_node(start)
            G.add_node(end)
            G.add_edge(start, end, line_index=i, geometry=line)
        
        return G
    
    def _find_cycles_in_network(self, graph: nx.Graph) -> List[List[Tuple]]:
        """Find cycles in the line network that could represent rooms"""
        cycles = []
        
        try:
            # Find all simple cycles
            simple_cycles = list(nx.simple_cycles(graph.to_directed()))
            
            # Filter cycles by size and geometric properties
            for cycle in simple_cycles:
                if 3 <= len(cycle) <= 20:  # Reasonable number of sides for a room
                    cycle_area = self._calculate_cycle_area(cycle)
                    if self.min_area <= cycle_area <= self.max_area:
                        cycles.append(cycle)
        except:
            # Fallback: manual cycle detection
            cycles = self._manual_cycle_detection(graph)
        
        return cycles
    
    def _manual_cycle_detection(self, graph: nx.Graph) -> List[List[Tuple]]:
        """Manual cycle detection as fallback"""
        cycles = []
        visited_edges = set()
        
        for node in graph.nodes():
            if graph.degree(node) >= 2:
                paths = self._dfs_find_cycles(graph, node, node, [], visited_edges)
                cycles.extend(paths)
        
        return cycles
    
    def _dfs_find_cycles(self, graph, start, current, path, visited_edges, max_depth=10):
        """Depth-first search to find cycles"""
        if len(path) > max_depth:
            return []
        
        cycles = []
        path.append(current)
        
        for neighbor in graph.neighbors(current):
            edge = tuple(sorted([current, neighbor]))
            
            if neighbor == start and len(path) > 3:
                # Found a cycle
                cycles.append(path[:])
            elif neighbor not in path and edge not in visited_edges:
                visited_edges.add(edge)
                cycles.extend(self._dfs_find_cycles(graph, start, neighbor, path, visited_edges, max_depth))
                visited_edges.remove(edge)
        
        path.pop()
        return cycles
    
    def _create_zone_from_cycle(self, cycle: List[Tuple], graph: nx.Graph) -> Dict[str, Any]:
        """Create a zone from a detected cycle"""
        try:
            # Create polygon from cycle points
            polygon = Polygon(cycle)
            
            if polygon.is_valid and polygon.area >= self.min_area:
                points = list(polygon.exterior.coords)
                
                return {
                    'points': points[:-1],  # Remove duplicate last point
                    'area': polygon.area,
                    'perimeter': polygon.length,
                    'bounds': list(polygon.bounds),
                    'zone_type': 'line_network',
                    'entity_type': 'DETECTED_ZONE',
                    'layer': 'AUTO_DETECTED',
                    'centroid': list(polygon.centroid.coords[0])
                }
        except Exception as e:
            print(f"Error creating zone from cycle: {e}")
        
        return None
    
    def _create_zone_from_points(self, points: List[Tuple], source_entity: Dict) -> Dict[str, Any]:
        """Create a zone from a list of points"""
        try:
            if len(points) < 3:
                return None
            
            # Ensure polygon is closed
            if points[0] != points[-1]:
                points = points + [points[0]]
            
            polygon = Polygon(points[:-1])  # Exclude duplicate last point for Polygon creation
            
            if polygon.is_valid and polygon.area >= self.min_area:
                return {
                    'points': points[:-1],  # Remove duplicate last point
                    'area': polygon.area,
                    'perimeter': polygon.length,
                    'bounds': list(polygon.bounds),
                    'zone_type': 'direct_polygon',
                    'entity_type': source_entity.get('entity_type', 'UNKNOWN'),
                    'layer': source_entity.get('layer', '0'),
                    'centroid': list(polygon.centroid.coords[0])
                }
        except Exception as e:
            print(f"Error creating zone from points: {e}")
        
        return None
    
    def _is_polygon_closed(self, points: List[Tuple], tolerance: float = 0.1) -> bool:
        """Check if a polygon is closed within tolerance"""
        if len(points) < 3:
            return False
        
        start = Point(points[0])
        end = Point(points[-1])
        return start.distance(end) <= tolerance
    
    def _is_likely_room_label(self, text: str) -> bool:
        """Determine if text is likely a room label"""
        text = text.lower().strip()
        
        room_keywords = [
            'room', 'bedroom', 'kitchen', 'bathroom', 'living', 'office',
            'hall', 'corridor', 'closet', 'storage', 'lobby', 'entrance',
            'dining', 'family', 'study', 'utility', 'laundry', 'garage'
        ]
        
        # Check for room keywords
        for keyword in room_keywords:
            if keyword in text:
                return True
        
        # Check for room numbers (e.g., "101", "Room 1", "R1")
        if any(char.isdigit() for char in text) and len(text) <= 10:
            return True
        
        return False
    
    def _find_enclosing_geometry(self, point: Tuple, entities: List[Dict]) -> Dict[str, Any]:
        """Find geometry that encloses a given point"""
        point_geom = Point(point)
        
        for entity in entities:
            if entity.get('entity_type') in ['LWPOLYLINE', 'POLYLINE', 'HATCH']:
                points = entity.get('points', [])
                if len(points) >= 3:
                    try:
                        polygon = Polygon(points)
                        if polygon.is_valid and polygon.contains(point_geom):
                            return self._create_zone_from_points(points, entity)
                    except:
                        continue
        
        return None
    
    def _calculate_cycle_area(self, cycle: List[Tuple]) -> float:
        """Calculate area of a cycle using shoelace formula"""
        if len(cycle) < 3:
            return 0.0
        
        try:
            polygon = Polygon(cycle)
            return polygon.area
        except:
            return 0.0
    
    def _remove_duplicate_zones(self, zones: List[Dict]) -> List[Dict]:
        """Remove duplicate zones based on overlap"""
        unique_zones = []
        
        for zone in zones:
            is_duplicate = False
            zone_poly = Polygon(zone['points'])
            
            for existing_zone in unique_zones:
                existing_poly = Polygon(existing_zone['points'])
                
                # Check for significant overlap
                if zone_poly.intersects(existing_poly):
                    intersection_area = zone_poly.intersection(existing_poly).area
                    overlap_ratio = intersection_area / min(zone_poly.area, existing_poly.area)
                    
                    if overlap_ratio > 0.8:  # 80% overlap threshold
                        is_duplicate = True
                        # Keep the larger zone
                        if zone_poly.area > existing_poly.area:
                            unique_zones.remove(existing_zone)
                            unique_zones.append(zone)
                        break
            
            if not is_duplicate:
                unique_zones.append(zone)
        
        return unique_zones
    
    def _validate_zones(self, zones: List[Dict]) -> List[Dict]:
        """Validate zones based on geometric and architectural criteria"""
        validated_zones = []
        
        for zone in zones:
            # Check area bounds
            area = zone.get('area', 0)
            if not (self.min_area <= area <= self.max_area):
                continue
            
            # Check aspect ratio (not too elongated)
            bounds = zone.get('bounds', [])
            if len(bounds) == 4:
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = max(width, height) / max(min(width, height), 0.1)
                
                if aspect_ratio > 20:  # Too elongated
                    continue
            
            # Check polygon validity
            try:
                polygon = Polygon(zone['points'])
                if not polygon.is_valid:
                    continue
            except:
                continue
            
            validated_zones.append(zone)
        
        return validated_zones
    
    def _classify_zones(self, zones: List[Dict]) -> List[Dict]:
        """Classify zones by likely room type based on geometry"""
        for zone in zones:
            area = zone.get('area', 0)
            room_label = zone.get('room_label', '').lower()
            
            # Basic classification based on area and shape
            if room_label:
                # Use explicit room label if available
                if 'kitchen' in room_label:
                    zone['likely_room_type'] = 'Kitchen'
                elif 'bathroom' in room_label or 'bath' in room_label:
                    zone['likely_room_type'] = 'Bathroom'
                elif 'bedroom' in room_label or 'bed' in room_label:
                    zone['likely_room_type'] = 'Bedroom'
                elif 'living' in room_label:
                    zone['likely_room_type'] = 'Living Room'
                elif 'office' in room_label:
                    zone['likely_room_type'] = 'Office'
                else:
                    zone['likely_room_type'] = 'Room'
            else:
                # Classify based on area
                if area < 8:
                    zone['likely_room_type'] = 'Bathroom'
                elif area < 15:
                    zone['likely_room_type'] = 'Bedroom'
                elif area < 25:
                    zone['likely_room_type'] = 'Kitchen'
                elif area < 50:
                    zone['likely_room_type'] = 'Living Room'
                else:
                    zone['likely_room_type'] = 'Large Space'
            
            # Add confidence score
            if room_label:
                zone['classification_confidence'] = 0.9
            else:
                zone['classification_confidence'] = 0.6
        
        return zones