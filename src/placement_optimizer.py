"""
Advanced Placement Optimizer for Furniture and Equipment
Handles intelligent placement of items within architectural spaces
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import random
import math


class PlacementOptimizer:
    """Advanced optimizer for furniture and equipment placement"""
    
    def __init__(self):
        self.placement_types = {
            'kitchen_island': {
                'typical_size': [3.0, 1.5],
                'min_clearance': 1.0,
                'preferred_position': 'center',
                'orientation_flexible': True
            },
            'workstation': {
                'typical_size': [1.5, 0.8],
                'min_clearance': 0.8,
                'preferred_position': 'wall_adjacent',
                'orientation_flexible': False
            },
            'equipment_rack': {
                'typical_size': [0.6, 1.0],
                'min_clearance': 0.5,
                'preferred_position': 'corner',
                'orientation_flexible': True
            },
            'conference_table': {
                'typical_size': [4.0, 1.5],
                'min_clearance': 1.2,
                'preferred_position': 'center',
                'orientation_flexible': True
            }
        }
    
    def optimize_placements(self, zones: List[Dict], parameters: Dict) -> Dict[str, Any]:
        """
        Optimize placement of items across all zones
        
        Args:
            zones: List of zone dictionaries with geometric data
            parameters: Placement parameters including item size, constraints
            
        Returns:
            Dictionary with placement results and optimization metrics
        """
        placement_results = {}
        total_items_placed = 0
        total_area_utilized = 0.0
        placement_efficiency_scores = []
        
        # Extract parameters
        item_length = parameters.get('box_length', 2.0)
        item_width = parameters.get('box_width', 1.5)
        margin = parameters.get('margin', 0.5)
        allow_rotation = parameters.get('enable_rotation', True)
        smart_spacing = parameters.get('smart_spacing', True)
        placement_type = parameters.get('placement_type', 'kitchen_island')
        
        # Get placement configuration
        config = self.placement_types.get(placement_type, self.placement_types['kitchen_island'])
        min_clearance = max(margin, config['min_clearance'])
        
        for i, zone in enumerate(zones):
            zone_name = f"Zone_{i+1}"
            
            # Create polygon from zone points
            try:
                if len(zone.get('points', [])) < 3:
                    continue
                    
                zone_polygon = Polygon(zone['points'])
                if not zone_polygon.is_valid or zone_polygon.area < 1.0:
                    continue
                
                # Optimize placement for this zone
                zone_placements = self._optimize_zone_placement(
                    zone_polygon=zone_polygon,
                    item_size=[item_length, item_width],
                    margin=min_clearance,
                    allow_rotation=allow_rotation,
                    smart_spacing=smart_spacing,
                    placement_config=config,
                    zone_data=zone
                )
                
                if zone_placements:
                    placement_results[zone_name] = zone_placements
                    total_items_placed += len(zone_placements)
                    
                    # Calculate area utilization
                    item_area = item_length * item_width
                    utilized_area = len(zone_placements) * item_area
                    total_area_utilized += utilized_area
                    
                    # Calculate efficiency score for this zone
                    zone_efficiency = min(1.0, utilized_area / (zone_polygon.area * 0.7))  # Max 70% utilization
                    placement_efficiency_scores.append(zone_efficiency)
                
            except Exception as e:
                print(f"Error optimizing zone {zone_name}: {e}")
                continue
        
        # Calculate overall metrics
        total_zone_area = sum(Polygon(zone['points']).area for zone in zones 
                             if len(zone.get('points', [])) >= 3)
        
        overall_efficiency = (
            sum(placement_efficiency_scores) / len(placement_efficiency_scores)
            if placement_efficiency_scores else 0.0
        )
        
        space_utilization = (
            total_area_utilized / total_zone_area
            if total_zone_area > 0 else 0.0
        )
        
        return {
            'placements': placement_results,
            'total_items': total_items_placed,
            'total_area_utilized': total_area_utilized,
            'space_utilization': min(1.0, space_utilization),
            'placement_efficiency': overall_efficiency,
            'optimization_details': {
                'algorithm': 'Advanced Grid + Smart Positioning',
                'placement_type': placement_type,
                'item_dimensions': [item_length, item_width],
                'clearance_used': min_clearance,
                'rotation_enabled': allow_rotation,
                'smart_spacing_enabled': smart_spacing
            }
        }
    
    def _optimize_zone_placement(self, zone_polygon: Polygon, item_size: List[float], 
                                margin: float, allow_rotation: bool, smart_spacing: bool,
                                placement_config: Dict, zone_data: Dict) -> List[Dict]:
        """Optimize placement within a single zone"""
        
        placements = []
        item_length, item_width = item_size
        
        # Get zone bounds
        minx, miny, maxx, maxy = zone_polygon.bounds
        
        # Try different placement strategies based on room type and configuration
        preferred_position = placement_config.get('preferred_position', 'center')
        
        if preferred_position == 'center':
            placements.extend(self._center_placement_strategy(
                zone_polygon, item_size, margin, allow_rotation
            ))
        elif preferred_position == 'wall_adjacent':
            placements.extend(self._wall_adjacent_strategy(
                zone_polygon, item_size, margin, allow_rotation
            ))
        elif preferred_position == 'corner':
            placements.extend(self._corner_placement_strategy(
                zone_polygon, item_size, margin, allow_rotation
            ))
        
        # If no placements found with preferred strategy, try grid placement
        if not placements:
            placements.extend(self._grid_placement_strategy(
                zone_polygon, item_size, margin, allow_rotation, smart_spacing
            ))
        
        # Validate and optimize placements
        validated_placements = self._validate_placements(placements, zone_polygon, margin)
        
        # Add metadata to placements
        for i, placement in enumerate(validated_placements):
            placement.update({
                'id': f"{zone_data.get('layer', 'zone')}_{i+1}",
                'placement_strategy': preferred_position,
                'clearance': margin,
                'zone_area': zone_polygon.area,
                'accessibility_score': self._calculate_accessibility_score(
                    placement, zone_polygon, validated_placements
                )
            })
        
        return validated_placements
    
    def _center_placement_strategy(self, zone_polygon: Polygon, item_size: List[float], 
                                  margin: float, allow_rotation: bool) -> List[Dict]:
        """Place items starting from center of the zone"""
        placements = []
        item_length, item_width = item_size
        
        # Get zone centroid
        centroid = zone_polygon.centroid
        cx, cy = centroid.x, centroid.y
        
        # Try placing one item at center first
        center_placement = self._try_placement_at_position(
            cx, cy, item_length, item_width, zone_polygon, margin, allow_rotation
        )
        
        if center_placement:
            placements.append(center_placement)
            
            # Try placing additional items around the center
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                for distance in [item_length + margin, item_length * 1.5 + margin]:
                    offset_x = distance * math.cos(math.radians(angle))
                    offset_y = distance * math.sin(math.radians(angle))
                    
                    new_x = cx + offset_x
                    new_y = cy + offset_y
                    
                    placement = self._try_placement_at_position(
                        new_x, new_y, item_length, item_width, zone_polygon, margin, allow_rotation
                    )
                    
                    if placement and not self._overlaps_existing(placement, placements, margin):
                        placements.append(placement)
        
        return placements
    
    def _wall_adjacent_strategy(self, zone_polygon: Polygon, item_size: List[float], 
                               margin: float, allow_rotation: bool) -> List[Dict]:
        """Place items adjacent to walls"""
        placements = []
        item_length, item_width = item_size
        
        # Get zone boundary
        boundary = zone_polygon.boundary
        
        # Sample points along the boundary
        boundary_length = boundary.length
        sample_distance = min(item_length, item_width) / 2
        
        for distance in np.arange(0, boundary_length, sample_distance):
            point = boundary.interpolate(distance)
            px, py = point.x, point.y
            
            # Calculate inward normal direction
            next_point = boundary.interpolate(distance + 0.1)
            if next_point:
                dx = next_point.x - px
                dy = next_point.y - py
                # Perpendicular inward direction
                normal_x = -dy
                normal_y = dx
                length = math.sqrt(normal_x**2 + normal_y**2)
                if length > 0:
                    normal_x /= length
                    normal_y /= length
                
                # Place item inward from wall
                offset_distance = max(item_width, item_length) / 2 + margin
                new_x = px + normal_x * offset_distance
                new_y = py + normal_y * offset_distance
                
                placement = self._try_placement_at_position(
                    new_x, new_y, item_length, item_width, zone_polygon, margin, allow_rotation
                )
                
                if placement and not self._overlaps_existing(placement, placements, margin):
                    placements.append(placement)
        
        return placements
    
    def _corner_placement_strategy(self, zone_polygon: Polygon, item_size: List[float], 
                                  margin: float, allow_rotation: bool) -> List[Dict]:
        """Place items in corners of the zone"""
        placements = []
        item_length, item_width = item_size
        
        # Find corner-like regions by analyzing polygon vertices
        coords = list(zone_polygon.exterior.coords[:-1])  # Exclude duplicate last point
        
        for i, (x, y) in enumerate(coords):
            # Calculate corner angle
            prev_idx = (i - 1) % len(coords)
            next_idx = (i + 1) % len(coords)
            
            prev_x, prev_y = coords[prev_idx]
            next_x, next_y = coords[next_idx]
            
            # Vectors from current point
            v1 = [prev_x - x, prev_y - y]
            v2 = [next_x - x, next_y - y]
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                angle = math.acos(max(-1, min(1, cos_angle)))
                
                # If it's a corner (angle < 120 degrees), try placement
                if angle < math.pi * 2/3:
                    # Calculate inward direction
                    bisector_x = (v1[0]/mag1 + v2[0]/mag2) / 2
                    bisector_y = (v1[1]/mag1 + v2[1]/mag2) / 2
                    bisector_mag = math.sqrt(bisector_x**2 + bisector_y**2)
                    
                    if bisector_mag > 0:
                        bisector_x /= bisector_mag
                        bisector_y /= bisector_mag
                        
                        # Place item inward from corner
                        offset_distance = max(item_width, item_length) / 2 + margin
                        new_x = x + bisector_x * offset_distance
                        new_y = y + bisector_y * offset_distance
                        
                        placement = self._try_placement_at_position(
                            new_x, new_y, item_length, item_width, zone_polygon, margin, allow_rotation
                        )
                        
                        if placement and not self._overlaps_existing(placement, placements, margin):
                            placements.append(placement)
        
        return placements
    
    def _grid_placement_strategy(self, zone_polygon: Polygon, item_size: List[float], 
                                margin: float, allow_rotation: bool, smart_spacing: bool) -> List[Dict]:
        """Grid-based placement strategy"""
        placements = []
        item_length, item_width = item_size
        
        # Get zone bounds
        minx, miny, maxx, maxy = zone_polygon.bounds
        
        # Calculate grid spacing
        spacing_x = item_length + margin
        spacing_y = item_width + margin
        
        if smart_spacing:
            # Adjust spacing based on zone size
            zone_width = maxx - minx
            zone_height = maxy - miny
            
            # Optimize grid to fit zone better
            cols = max(1, int(zone_width / spacing_x))
            rows = max(1, int(zone_height / spacing_y))
            
            if cols > 1:
                spacing_x = zone_width / cols
            if rows > 1:
                spacing_y = zone_height / rows
        
        # Generate grid points
        y = miny + spacing_y/2
        while y < maxy:
            x = minx + spacing_x/2
            while x < maxx:
                placement = self._try_placement_at_position(
                    x, y, item_length, item_width, zone_polygon, margin, allow_rotation
                )
                
                if placement and not self._overlaps_existing(placement, placements, margin):
                    placements.append(placement)
                
                x += spacing_x
            y += spacing_y
        
        return placements
    
    def _try_placement_at_position(self, x: float, y: float, length: float, width: float,
                                  zone_polygon: Polygon, margin: float, allow_rotation: bool) -> Optional[Dict]:
        """Try to place an item at a specific position"""
        
        orientations = [(length, width)]
        if allow_rotation and abs(length - width) > 0.1:  # Only rotate if dimensions differ
            orientations.append((width, length))
        
        for item_w, item_h in orientations:
            # Create item rectangle
            item_box = box(x - item_w/2, y - item_h/2, x + item_w/2, y + item_h/2)
            
            # Check if item fits within zone with margin
            buffered_zone = zone_polygon.buffer(-margin) if margin > 0 else zone_polygon
            
            if buffered_zone.is_valid and buffered_zone.contains(item_box):
                return {
                    'position': [x, y],
                    'size': [item_w, item_h],
                    'rotation': 0 if (item_w, item_h) == (length, width) else 90,
                    'area': item_w * item_h,
                    'bounds': [x - item_w/2, y - item_h/2, x + item_w/2, y + item_h/2]
                }
        
        return None
    
    def _overlaps_existing(self, new_placement: Dict, existing_placements: List[Dict], margin: float) -> bool:
        """Check if new placement overlaps with existing ones"""
        new_bounds = new_placement['bounds']
        new_box = box(new_bounds[0] - margin, new_bounds[1] - margin, 
                     new_bounds[2] + margin, new_bounds[3] + margin)
        
        for existing in existing_placements:
            existing_bounds = existing['bounds']
            existing_box = box(existing_bounds[0], existing_bounds[1], 
                             existing_bounds[2], existing_bounds[3])
            
            if new_box.intersects(existing_box):
                return True
        
        return False
    
    def _validate_placements(self, placements: List[Dict], zone_polygon: Polygon, margin: float) -> List[Dict]:
        """Validate and filter placements"""
        validated = []
        
        for placement in placements:
            bounds = placement['bounds']
            item_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
            
            # Check if still within zone
            if zone_polygon.contains(item_box) or zone_polygon.intersects(item_box):
                # Check intersection area is significant
                intersection = zone_polygon.intersection(item_box)
                if intersection.area > item_box.area * 0.8:  # 80% must be within zone
                    validated.append(placement)
        
        return validated
    
    def _calculate_accessibility_score(self, placement: Dict, zone_polygon: Polygon, 
                                     all_placements: List[Dict]) -> float:
        """Calculate accessibility score for a placement"""
        pos_x, pos_y = placement['position']
        
        # Distance to zone center
        centroid = zone_polygon.centroid
        center_distance = math.sqrt((pos_x - centroid.x)**2 + (pos_y - centroid.y)**2)
        
        # Distance to zone boundary
        boundary_distance = zone_polygon.boundary.distance(Point(pos_x, pos_y))
        
        # Distance to other placements
        min_neighbor_distance = float('inf')
        for other in all_placements:
            if other != placement:
                other_x, other_y = other['position']
                distance = math.sqrt((pos_x - other_x)**2 + (pos_y - other_y)**2)
                min_neighbor_distance = min(min_neighbor_distance, distance)
        
        # Calculate score (higher is better)
        # Prefer moderate distance from center, good distance from boundary and neighbors
        center_score = 1.0 / (1.0 + center_distance / 10.0)
        boundary_score = min(1.0, boundary_distance / 2.0)
        neighbor_score = min(1.0, min_neighbor_distance / 3.0) if min_neighbor_distance != float('inf') else 1.0
        
        return (center_score + boundary_score + neighbor_score) / 3.0