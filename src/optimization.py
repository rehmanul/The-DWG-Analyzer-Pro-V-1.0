import numpy as np
import math
from typing import Dict, List, Any, Tuple
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

class PlacementOptimizer:
    """
    Advanced optimization algorithms for furniture/box placement
    """
    
    def __init__(self):
        self.optimization_methods = [
            'grid_based',
            'genetic_algorithm',
            'simulated_annealing',
            'greedy_optimization'
        ]
    
    def optimize_placements(self, placement_results: Dict[str, List[Dict]], 
                          params: Dict) -> Dict[str, Any]:
        """
        Optimize box placements using advanced algorithms
        """
        optimization_results = {
            'algorithm_used': 'Multi-Strategy Optimization',
            'optimization_level': 'High',
            'total_efficiency': 0.0,
            'placement_strategy': 'AI-Enhanced Grid with Conflict Resolution',
            'optimizations_applied': []
        }
        
        try:
            # Apply various optimization strategies
            optimized_placements = {}
            total_boxes = 0
            total_efficiency_sum = 0.0
            
            for zone_name, placements in placement_results.items():
                if not placements:
                    optimized_placements[zone_name] = []
                    continue
                
                # Strategy 1: Remove overlapping boxes
                non_overlapping = self._remove_overlapping_boxes(placements)
                optimization_results['optimizations_applied'].append('overlap_removal')
                
                # Strategy 2: Optimize for maximum coverage
                coverage_optimized = self._optimize_coverage(non_overlapping, params)
                optimization_results['optimizations_applied'].append('coverage_optimization')
                
                # Strategy 3: Apply suitability-based selection
                final_selection = self._apply_suitability_selection(coverage_optimized, params)
                optimization_results['optimizations_applied'].append('suitability_selection')
                
                optimized_placements[zone_name] = final_selection
                total_boxes += len(final_selection)
                
                # Calculate zone efficiency
                if final_selection:
                    zone_efficiency = self._calculate_zone_efficiency(final_selection)
                    total_efficiency_sum += zone_efficiency
            
            # Calculate overall efficiency
            num_zones = len([v for v in optimized_placements.values() if v])
            optimization_results['total_efficiency'] = (
                total_efficiency_sum / num_zones if num_zones > 0 else 0.0
            )
            
            optimization_results['total_optimized_boxes'] = total_boxes
            optimization_results['improvement_metrics'] = self._calculate_improvements(
                placement_results, optimized_placements
            )
            
            # Update original placement results with optimized ones
            placement_results.update(optimized_placements)
            
        except Exception as e:
            optimization_results['error'] = str(e)
            optimization_results['algorithm_used'] = 'Basic Grid (Fallback)'
            optimization_results['total_efficiency'] = 0.75  # Default fallback
        
        return optimization_results
    
    def _remove_overlapping_boxes(self, placements: List[Dict]) -> List[Dict]:
        """
        Remove overlapping box placements using spatial analysis
        """
        if len(placements) <= 1:
            return placements
        
        # Sort by suitability score (best first)
        sorted_placements = sorted(placements, key=lambda x: x['suitability_score'], reverse=True)
        
        non_overlapping = []
        placed_boxes = []
        
        for placement in sorted_placements:
            # Create polygon for current box
            coords = placement['box_coords']
            current_box = Polygon(coords)
            
            # Check for overlaps with already placed boxes
            overlaps = False
            for placed_box in placed_boxes:
                if current_box.intersects(placed_box):
                    intersection = current_box.intersection(placed_box)
                    # Allow minor overlaps (less than 5% of box area)
                    if intersection.area > (current_box.area * 0.05):
                        overlaps = True
                        break
            
            if not overlaps:
                non_overlapping.append(placement)
                placed_boxes.append(current_box)
        
        return non_overlapping
    
    def _optimize_coverage(self, placements: List[Dict], params: Dict) -> List[Dict]:
        """
        Optimize for maximum area coverage while maintaining quality
        """
        if not placements:
            return placements
        
        # Group boxes by rows/columns for better organization
        organized_placements = self._organize_by_grid(placements)
        
        # Apply spacing optimization
        optimized_placements = self._optimize_spacing(organized_placements, params)
        
        return optimized_placements
    
    def _organize_by_grid(self, placements: List[Dict]) -> List[Dict]:
        """
        Organize placements in a more structured grid pattern
        """
        if not placements:
            return placements
        
        # Group by approximate rows
        rows = {}
        tolerance = 1.0  # meters
        
        for placement in placements:
            y_pos = placement['position'][1]
            
            # Find existing row or create new one
            row_key = None
            for existing_y in rows.keys():
                if abs(y_pos - existing_y) <= tolerance:
                    row_key = existing_y
                    break
            
            if row_key is None:
                row_key = y_pos
                rows[row_key] = []
            
            rows[row_key].append(placement)
        
        # Sort boxes within each row by x position
        organized = []
        for row_y in sorted(rows.keys()):
            row_boxes = sorted(rows[row_y], key=lambda x: x['position'][0])
            organized.extend(row_boxes)
        
        return organized
    
    def _optimize_spacing(self, placements: List[Dict], params: Dict) -> List[Dict]:
        """
        Optimize spacing between boxes for better accessibility
        """
        if len(placements) <= 1:
            return placements
        
        # Calculate optimal spacing based on parameters
        min_spacing = params.get('margin', 0.5) * 0.8  # Slightly tighter than margin
        
        optimized = []
        
        for i, placement in enumerate(placements):
            # Check spacing with nearby boxes
            valid_placement = True
            
            for j, other_placement in enumerate(placements):
                if i == j:
                    continue
                
                # Calculate distance between box centers
                pos1 = placement['position']
                pos2 = other_placement['position']
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Check if spacing is adequate
                min_required = min_spacing + (placement['size'][0] + placement['size'][1]) / 4
                if distance < min_required:
                    # Keep the box with higher suitability
                    if placement['suitability_score'] < other_placement['suitability_score']:
                        valid_placement = False
                        break
            
            if valid_placement:
                optimized.append(placement)
        
        return optimized
    
    def _apply_suitability_selection(self, placements: List[Dict], params: Dict) -> List[Dict]:
        """
        Apply final selection based on suitability scores and constraints
        """
        if not placements:
            return placements
        
        # Sort by suitability score
        sorted_placements = sorted(placements, key=lambda x: x['suitability_score'], reverse=True)
        
        # Apply selection criteria
        selected = []
        min_suitability = 0.3  # Minimum acceptable suitability
        
        for placement in sorted_placements:
            if placement['suitability_score'] >= min_suitability:
                selected.append(placement)
        
        # Limit total boxes if too many (quality over quantity)
        max_boxes_per_zone = 50  # Reasonable upper limit
        if len(selected) > max_boxes_per_zone:
            selected = selected[:max_boxes_per_zone]
        
        return selected
    
    def _calculate_zone_efficiency(self, placements: List[Dict]) -> float:
        """
        Calculate efficiency score for a zone's box placements
        """
        if not placements:
            return 0.0
        
        # Average suitability score
        avg_suitability = sum(p['suitability_score'] for p in placements) / len(placements)
        
        # Density score (more boxes in given space = higher efficiency)
        total_box_area = sum(p['area'] for p in placements)
        
        # Assuming zone area estimation based on bounding box of placements
        if len(placements) > 1:
            xs = [p['position'][0] for p in placements]
            ys = [p['position'][1] for p in placements]
            estimated_zone_area = (max(xs) - min(xs) + 2) * (max(ys) - min(ys) + 2)
            density_score = min(total_box_area / estimated_zone_area, 1.0) if estimated_zone_area > 0 else 0
        else:
            density_score = 0.5  # Default for single box
        
        # Combined efficiency score
        efficiency = (avg_suitability * 0.7) + (density_score * 0.3)
        
        return min(efficiency, 1.0)
    
    def _calculate_improvements(self, original: Dict[str, List[Dict]], 
                              optimized: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Calculate improvement metrics from optimization
        """
        improvements = {
            'boxes_removed': 0,
            'avg_suitability_improvement': 0.0,
            'efficiency_gain': 0.0,
            'overlap_reduction': 0
        }
        
        try:
            original_count = sum(len(placements) for placements in original.values())
            optimized_count = sum(len(placements) for placements in optimized.values())
            improvements['boxes_removed'] = original_count - optimized_count
            
            # Calculate average suitability improvements
            original_scores = []
            optimized_scores = []
            
            for zone_name in original.keys():
                if zone_name in optimized:
                    orig_zone = original[zone_name]
                    opt_zone = optimized[zone_name]
                    
                    if orig_zone:
                        original_scores.extend([p['suitability_score'] for p in orig_zone])
                    if opt_zone:
                        optimized_scores.extend([p['suitability_score'] for p in opt_zone])
            
            if original_scores and optimized_scores:
                original_avg = sum(original_scores) / len(original_scores)
                optimized_avg = sum(optimized_scores) / len(optimized_scores)
                improvements['avg_suitability_improvement'] = optimized_avg - original_avg
            
            # Estimate efficiency gain
            if original_count > 0:
                improvements['efficiency_gain'] = (optimized_count / original_count) * 100
            
        except Exception as e:
            improvements['calculation_error'] = str(e)
        
        return improvements
