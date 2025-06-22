import numpy as np
import math
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from typing import List, Dict, Tuple, Any

class AIAnalyzer:
    """
    AI-powered analyzer for architectural space analysis and room type detection
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
        # Room type classification parameters
        self.room_classifications = {
            'corridor': {'min_aspect_ratio': 3.0, 'max_area': 20},
            'storage_wc': {'max_area': 8, 'max_dimension': 3},
            'small_office': {'min_area': 5, 'max_area': 15, 'max_aspect_ratio': 2.0},
            'office': {'min_area': 10, 'max_area': 35, 'max_aspect_ratio': 2.5},
            'meeting_room': {'min_area': 15, 'max_area': 40, 'max_aspect_ratio': 1.8},
            'conference_room': {'min_area': 30, 'max_area': 80, 'max_aspect_ratio': 1.5},
            'open_office': {'min_area': 35, 'max_aspect_ratio': 3.0},
            'hall_auditorium': {'min_area': 70}
        }
    
    def analyze_room_types(self, zones: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze room types using AI-powered geometric analysis
        """
        room_analysis = {}
        
        for i, zone in enumerate(zones):
            if not zone.get('points'):
                continue
            
            try:
                # Create polygon from points
                poly = Polygon(zone['points'])
                if not poly.is_valid:
                    poly = poly.buffer(0)  # Fix invalid polygons
                
                # Calculate geometric properties
                area = poly.area
                bounds = poly.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                perimeter = poly.length
                
                # Calculate derived metrics
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
                compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # AI-powered room classification
                room_type, confidence = self._classify_room(area, width, height, aspect_ratio, compactness)
                
                room_analysis[f"Zone_{i}"] = {
                    'type': room_type,
                    'confidence': confidence,
                    'area': area,
                    'dimensions': (width, height),
                    'aspect_ratio': aspect_ratio,
                    'compactness': compactness,
                    'perimeter': perimeter,
                    'layer': zone.get('layer', 'Unknown'),
                    'center': poly.centroid.coords[0] if poly.centroid else (0, 0)
                }
                
            except Exception as e:
                # Handle invalid geometries
                room_analysis[f"Zone_{i}"] = {
                    'type': 'Invalid Geometry',
                    'confidence': 0.0,
                    'area': 0,
                    'dimensions': (0, 0),
                    'aspect_ratio': 1,
                    'compactness': 0,
                    'perimeter': 0,
                    'layer': zone.get('layer', 'Unknown'),
                    'center': (0, 0),
                    'error': str(e)
                }
        
        return room_analysis
    
    def _classify_room(self, area: float, width: float, height: float, 
                      aspect_ratio: float, compactness: float) -> Tuple[str, float]:
        """
        Classify room type using AI heuristics
        """
        # Initialize with unknown
        best_type = "Unknown"
        best_confidence = 0.3
        
        # Rule-based classification with confidence scoring
        rules = []
        
        # Corridor detection
        if aspect_ratio > 3.0 and area < 25:
            rules.append(("Corridor", 0.9 * min(aspect_ratio / 3.0, 2.0)))
        
        # Storage/WC detection
        if area < 8 and max(width, height) < 3:
            rules.append(("Storage/WC", 0.85))
        elif area < 5:
            rules.append(("Storage/WC", 0.7))
        
        # Small office
        if 5 <= area < 15 and aspect_ratio < 2.0:
            confidence = 0.8 - abs(aspect_ratio - 1.2) * 0.2
            rules.append(("Small Office", max(confidence, 0.6)))
        
        # Regular office
        if 10 <= area < 35 and aspect_ratio < 2.5:
            confidence = 0.75 - abs(aspect_ratio - 1.5) * 0.1
            rules.append(("Office", max(confidence, 0.6)))
        
        # Meeting room (more square, medium size)
        if 15 <= area < 40 and aspect_ratio < 1.8 and compactness > 0.6:
            confidence = 0.85 - abs(aspect_ratio - 1.2) * 0.1
            rules.append(("Meeting Room", max(confidence, 0.7)))
        
        # Conference room (larger, still relatively square)
        if 30 <= area < 80 and aspect_ratio < 1.5:
            confidence = 0.8 + compactness * 0.1
            rules.append(("Conference Room", min(confidence, 0.9)))
        
        # Open office (large, can be elongated)
        if area >= 35 and aspect_ratio < 3.0:
            confidence = 0.7 + min((area - 35) / 100, 0.2)
            rules.append(("Open Office", min(confidence, 0.85)))
        
        # Hall/Auditorium (very large)
        if area >= 70:
            confidence = 0.8 + min((area - 70) / 200, 0.15)
            rules.append(("Hall/Auditorium", min(confidence, 0.95)))
        
        # Find best match
        for room_type, confidence in rules:
            if confidence > best_confidence:
                best_type = room_type
                best_confidence = confidence
        
        return best_type, best_confidence
    
    def analyze_furniture_placement(self, zones: List[Dict], params: Dict) -> Dict[str, List[Dict]]:
        """
        Analyze optimal furniture/box placement using AI algorithms
        """
        placement_results = {}
        
        box_size = params['box_size']
        margin = params['margin']
        allow_rotation = params.get('allow_rotation', True)
        smart_spacing = params.get('smart_spacing', True)
        
        for i, zone in enumerate(zones):
            if not zone.get('points'):
                continue
            
            try:
                poly = Polygon(zone['points'])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                placements = self._calculate_optimal_placements(
                    poly, box_size, margin, allow_rotation, smart_spacing
                )
                
                placement_results[f"Zone_{i}"] = placements
                
            except Exception as e:
                placement_results[f"Zone_{i}"] = []
        
        return placement_results
    
    def _calculate_optimal_placements(self, poly: Polygon, box_size: Tuple[float, float], 
                                    margin: float, allow_rotation: bool, 
                                    smart_spacing: bool) -> List[Dict]:
        """
        Calculate optimal box placements within a polygon
        """
        placements = []
        bounds = poly.bounds
        
        # Adaptive spacing based on room size and smart_spacing setting
        area = poly.area
        if smart_spacing:
            if area < 15:
                spacing_factor = 0.8  # Tighter spacing for small rooms
            elif area > 60:
                spacing_factor = 1.2  # More generous spacing for large rooms
            else:
                spacing_factor = 1.0
        else:
            spacing_factor = 1.0
        
        # Try different orientations
        orientations = [(box_size[0], box_size[1])]
        if allow_rotation:
            orientations.append((box_size[1], box_size[0]))
        
        best_placements = []
        
        for width, height in orientations:
            current_placements = []
            
            # Calculate step sizes
            x_step = (width + margin) * spacing_factor
            y_step = (height + margin) * spacing_factor
            
            # Grid-based placement with suitability scoring
            x = bounds[0] + margin
            while x + width <= bounds[2] - margin:
                y = bounds[1] + margin
                while y + height <= bounds[3] - margin:
                    # Create test box
                    test_box = box(x, y, x + width, y + height)
                    
                    # Check if box fits completely within polygon
                    if poly.contains(test_box):
                        suitability = self._calculate_suitability_score(poly, test_box)
                        
                        # Only include placements above minimum threshold
                        if suitability > 0.2:
                            current_placements.append({
                                'position': (x, y),
                                'size': (width, height),
                                'box_coords': [
                                    (x, y), (x + width, y),
                                    (x + width, y + height), (x, y + height)
                                ],
                                'suitability_score': suitability,
                                'area': width * height,
                                'orientation': 'original' if width == box_size[0] else 'rotated'
                            })
                    
                    y += y_step
                x += x_step
            
            # Keep the orientation that yields more placements
            if len(current_placements) > len(best_placements):
                best_placements = current_placements
        
        # Sort by suitability score (best first)
        best_placements.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return best_placements
    
    def _calculate_suitability_score(self, room_poly: Polygon, furniture_box: Polygon) -> float:
        """
        Calculate suitability score for furniture placement using multiple factors
        """
        center = furniture_box.centroid
        
        # Factor 1: Distance from walls (prefer some clearance)
        distance_to_boundary = room_poly.boundary.distance(center)
        wall_score = min(distance_to_boundary / 2.0, 1.0)
        
        # Factor 2: Distance from room center (prefer balanced distribution)
        room_center = room_poly.centroid
        distance_to_center = center.distance(room_center)
        room_radius = math.sqrt(room_poly.area / math.pi)
        center_score = max(0, 1.0 - (distance_to_center / room_radius))
        
        # Factor 3: Area utilization efficiency
        box_area = furniture_box.area
        room_area = room_poly.area
        utilization_score = min(box_area / (room_area * 0.15), 1.0)
        
        # Factor 4: Shape compatibility (prefer placement in regular areas)
        try:
            # Check local geometry around the box
            expanded_box = furniture_box.buffer(0.5)
            intersection_area = room_poly.intersection(expanded_box).area
            shape_score = intersection_area / expanded_box.area
        except:
            shape_score = 0.5
        
        # Weighted combination of factors
        total_score = (
            wall_score * 0.3 +
            center_score * 0.3 +
            utilization_score * 0.2 +
            shape_score * 0.2
        )
        
        return min(total_score, 1.0)
