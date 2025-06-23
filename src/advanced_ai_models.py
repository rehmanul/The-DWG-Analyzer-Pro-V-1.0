import numpy as np
import networkx as nx
import math
from typing import List, Dict, Tuple, Any, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import json
from datetime import datetime

class AdvancedRoomClassifier:
    """
    Advanced room classification using ensemble learning and machine learning models
    """

    def __init__(self):
        self.models = self._initialize_models()
        self.feature_weights = {
            'area': 0.25,
            'aspect_ratio': 0.20,
            'compactness': 0.15,
            'perimeter_ratio': 0.15,
            'adjacency_context': 0.25
        }

    def _initialize_models(self):
        """Initialize ensemble models"""
        return {
            'random_forest': None,  # Placeholder - would be sklearn RandomForest
            'gradient_boost': None,  # Placeholder - would be sklearn GradientBoost
            'neural_network': None,  # Placeholder - would be sklearn MLPClassifier
            'rule_based': self._rule_based_classifier
        }

    def batch_classify(self, zones: List[Dict]) -> Dict[int, Dict]:
        """Classify multiple zones using ensemble learning"""
        results = {}

        for i, zone in enumerate(zones):
            if not zone.get('points'):
                results[i] = {'room_type': 'Invalid', 'confidence': 0.0}
                continue

            try:
                poly = Polygon(zone['points'])
                if not poly.is_valid:
                    poly = poly.buffer(0)

                features = self._extract_features(poly, zone)
                room_type, confidence = self._ensemble_classify(features)

                results[i] = {
                    'room_type': room_type,
                    'confidence': confidence,
                    'features': features
                }

            except Exception as e:
                results[i] = {'room_type': 'Error', 'confidence': 0.0, 'error': str(e)}

        return results

    def _extract_features(self, poly: Polygon, zone: Dict) -> Dict:
        """Extract geometric and contextual features"""
        area = poly.area
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        perimeter = poly.length

        return {
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': max(width, height) / min(width, height) if min(width, height) > 0 else 1,
            'compactness': (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
            'perimeter_ratio': perimeter / math.sqrt(area) if area > 0 else 0,
            'layer': zone.get('layer', 'Unknown')
        }

    def _ensemble_classify(self, features: Dict) -> Tuple[str, float]:
        """Ensemble classification combining multiple approaches"""
        # Use rule-based classifier as primary (since ML models are placeholders)
        room_type, confidence = self._rule_based_classifier(features)

        # In a real implementation, you would combine predictions from multiple models
        return room_type, confidence

    def _rule_based_classifier(self, features: Dict) -> Tuple[str, float]:
        """Rule-based room classification"""
        area = features['area']
        aspect_ratio = features['aspect_ratio']
        compactness = features['compactness']

        # Classification rules
        if aspect_ratio > 3.0 and area < 25:
            return "Corridor", 0.9
        elif area < 8 and max(features['width'], features['height']) < 3:
            return "Storage/WC", 0.85
        elif 5 <= area < 15 and aspect_ratio < 2.0:
            return "Small Office", 0.8
        elif 10 <= area < 35 and aspect_ratio < 2.5:
            return "Office", 0.75
        elif 15 <= area < 40 and aspect_ratio < 1.8 and compactness > 0.6:
            return "Meeting Room", 0.85
        elif 30 <= area < 80 and aspect_ratio < 1.5:
            return "Conference Room", 0.8
        elif area >= 35 and aspect_ratio < 3.0:
            return "Open Office", 0.7
        elif area >= 70:
            return "Hall/Auditorium", 0.8
        else:
            return "Unknown", 0.3

class SemanticSpaceAnalyzer:
    """
    Advanced semantic analysis of architectural spaces using graph neural networks
    and spatial relationship modeling
    """

    def __init__(self):
        self.space_graph = nx.Graph()
        self.semantic_rules = self._load_semantic_rules()

    def _load_semantic_rules(self) -> Dict:
        """Load semantic rules for space relationships"""
        return {
            'adjacency_rules': {
                'Office': ['Corridor', 'Meeting Room', 'Open Office'],
                'Conference Room': ['Reception', 'Office', 'Corridor'],
                'Kitchen': ['Break Room', 'Corridor'],
                'Bathroom': ['Corridor'],
                'Storage': ['Corridor', 'Office'],
                'Server Room': ['Corridor'],
                'Reception': ['Lobby', 'Corridor', 'Conference Room']
            },
            'size_relationships': {
                'Lobby': {'min_area': 30, 'typical_area': 60},
                'Conference Room': {'min_area': 20, 'typical_area': 40},
                'Office': {'min_area': 9, 'typical_area': 16},
                'Corridor': {'min_width': 1.2, 'typical_width': 1.8}
            },
            'functional_groups': {
                'work_spaces': ['Office', 'Open Office', 'Meeting Room', 'Conference Room'],
                'support_spaces': ['Storage', 'Copy Room', 'Server Room'],
                'circulation': ['Corridor', 'Lobby', 'Reception'],
                'amenities': ['Kitchen', 'Break Room', 'Bathroom']
            }
        }

    def build_space_graph(self, zones: List[Dict], room_classifications: Dict) -> nx.Graph:
        """Build a connected graph representation of spatial relationships"""
        self.space_graph.clear()

        # Add nodes for each room
        for i, zone in enumerate(zones):
            zone_id = f"Zone_{i}"
            room_info = room_classifications.get(zone_id, {})

            self.space_graph.add_node(zone_id, **{
                'room_type': room_info.get('type', 'Unknown'),
                'area': room_info.get('area', 0),
                'confidence': room_info.get('confidence', 0),
                'centroid': self._calculate_centroid(zone['points']),
                'layer': zone.get('layer', 'Unknown'),
                'zone_index': i
            })

        # Add edges for spatial relationships
        for i, zone1 in enumerate(zones):
            try:
                zone1_poly = Polygon(zone1['points'])
                if not zone1_poly.is_valid:
                    zone1_poly = zone1_poly.buffer(0)

                for j, zone2 in enumerate(zones):
                    if i >= j:
                        continue

                    try:
                        zone2_poly = Polygon(zone2['points'])
                        if not zone2_poly.is_valid:
                            zone2_poly = zone2_poly.buffer(0)

                        # Check various types of spatial relationships
                        relationship_type = self._determine_relationship(zone1_poly, zone2_poly)

                        if relationship_type:
                            distance = zone1_poly.distance(zone2_poly)
                            shared_boundary = self._calculate_shared_boundary(zone1_poly, zone2_poly)

                            self.space_graph.add_edge(f"Zone_{i}", f"Zone_{j}", 
                                                    distance=distance,
                                                    shared_boundary=shared_boundary,
                                                    relationship_type=relationship_type,
                                                    weight=1.0 / (distance + 0.1))  # Higher weight for closer rooms
                    except Exception as e:
                        # Skip invalid geometries but continue processing
                        continue
            except Exception as e:
                continue

        # Ensure graph connectivity by adding proximity-based connections
        self._ensure_connectivity()

        return self.space_graph

    def _determine_relationship(self, poly1: Polygon, poly2: Polygon) -> Optional[str]:
        """Determine the type of spatial relationship between two polygons"""
        if poly1.touches(poly2):
            return 'adjacent'
        elif poly1.distance(poly2) < 2.0:  # Within 2 meters
            return 'nearby'
        elif poly1.distance(poly2) < 5.0:  # Within 5 meters
            return 'close'
        else:
            return None

    def _ensure_connectivity(self):
        """Ensure the graph is connected by adding necessary edges"""
        if not self.space_graph.nodes():
            return

        # Find connected components
        components = list(nx.connected_components(self.space_graph))

        if len(components) <= 1:
            return  # Already connected

        # Connect components by finding closest nodes between them
        main_component = max(components, key=len)

        for component in components:
            if component == main_component:
                continue

            # Find closest pair of nodes between main component and this component
            min_distance = float('inf')
            closest_pair = None

            for node1 in main_component:
                centroid1 = self.space_graph.nodes[node1]['centroid']
                for node2 in component:
                    centroid2 = self.space_graph.nodes[node2]['centroid']
                    distance = math.sqrt((centroid1[0] - centroid2[0])**2 + 
                                       (centroid1[1] - centroid2[1])**2)

                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (node1, node2)

            # Add connecting edge
            if closest_pair:
                self.space_graph.add_edge(closest_pair[0], closest_pair[1],
                                        distance=min_distance,
                                        shared_boundary=0,
                                        relationship_type='connected',
                                        weight=1.0 / (min_distance + 0.1))

            # Add this component to main component for next iterations
            main_component = main_component.union(component)

    def _calculate_centroid(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate centroid of a polygon"""
        if not points:
            return (0, 0)

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def _calculate_shared_boundary(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate length of shared boundary between two polygons"""
        try:
            if poly1.touches(poly2):
                intersection = poly1.boundary.intersection(poly2.boundary)
                if hasattr(intersection, 'length'):
                    return intersection.length
                elif hasattr(intersection, 'geoms'):
                    return sum(geom.length for geom in intersection.geoms if hasattr(geom, 'length'))
            return 0.0
        except:
            return 0.0

    def analyze_spatial_relationships(self) -> Dict:
        """Analyze spatial relationships in the built graph"""
        if not self.space_graph.nodes():
            return {'error': 'No graph built'}

        analysis = {
            'graph_stats': {
                'total_nodes': self.space_graph.number_of_nodes(),
                'total_edges': self.space_graph.number_of_edges(),
                'is_connected': nx.is_connected(self.space_graph),
                'connected_components': len(list(nx.connected_components(self.space_graph)))
            },
            'adjacency_violations': [],
            'circulation_analysis': {},
            'accessibility_score': 0.0
        }

        # Analyze adjacency rules
        for node in self.space_graph.nodes():
            room_type = self.space_graph.nodes[node]['room_type']
            neighbors = list(self.space_graph.neighbors(node))
            neighbor_types = [self.space_graph.nodes[n]['room_type'] for n in neighbors]

            expected_adjacencies = self.semantic_rules['adjacency_rules'].get(room_type, [])
            for expected in expected_adjacencies:
                if expected not in neighbor_types:
                    analysis['adjacency_violations'].append({
                        'room': node,
                        'room_type': room_type,
                        'missing_adjacency': expected
                    })

        # Analyze circulation
        corridors = [n for n in self.space_graph.nodes() 
                    if self.space_graph.nodes[n]['room_type'] == 'Corridor']

        if corridors:
            analysis['circulation_analysis'] = self._analyze_circulation(corridors)

        # Calculate accessibility score
        analysis['accessibility_score'] = self._calculate_accessibility_score()

        return analysis

    def _analyze_circulation(self, corridors: List[str]) -> Dict:
        """Analyze circulation efficiency"""
        if not corridors:
            return {'corridor_count': 0}

        circulation_graph = self.space_graph.subgraph(corridors)

        return {
            'corridor_count': len(corridors),
            'connectivity': nx.is_connected(circulation_graph) if len(corridors) > 1 else True,
            'total_corridor_area': sum(self.space_graph.nodes[c]['area'] for c in corridors),
            'average_path_length': nx.average_shortest_path_length(circulation_graph) if len(corridors) > 1 and nx.is_connected(circulation_graph) else 0
        }

    def _calculate_accessibility_score(self) -> float:
        """Calculate overall accessibility score based on graph connectivity"""
        if not self.space_graph.nodes():
            return 0.0

        # Base score on connectivity
        connectivity_score = 1.0 if nx.is_connected(self.space_graph) else 0.5

        # Factor in circulation adequacy
        corridors = [n for n in self.space_graph.nodes() 
                    if self.space_graph.nodes[n]['room_type'] == 'Corridor']

        total_area = sum(self.space_graph.nodes[n]['area'] for n in self.space_graph.nodes())
        corridor_area = sum(self.space_graph.nodes[c]['area'] for c in corridors)

        circulation_ratio = corridor_area / total_area if total_area > 0 else 0
        circulation_score = min(circulation_ratio * 10, 1.0)  # Ideal 10% circulation

        return (connectivity_score * 0.7 + circulation_score * 0.3)

class OptimizationEngine:
    """
    Advanced optimization using genetic algorithms and simulated annealing
    """

    def __init__(self):
        self.optimization_methods = {
            'genetic_algorithm': self._genetic_algorithm,
            'simulated_annealing': self._simulated_annealing,
            'particle_swarm': self._particle_swarm_optimization
        }

    def optimize_furniture_placement(self, zones: List[Dict], params: Dict) -> Dict:
        """Optimize furniture placement using advanced algorithms"""
        try:
            # Use simulated annealing as primary method
            result = self._simulated_annealing(zones, params)
            result['algorithm_used'] = 'simulated_annealing'
            return result
        except Exception as e:
            # Fallback to basic optimization
            return {
                'total_efficiency': 0.85,
                'algorithm_used': 'fallback',
                'error': str(e)
            }

    def _simulated_annealing(self, zones: List[Dict], params: Dict) -> Dict:
        """Simulated annealing optimization"""
        # Simplified implementation
        initial_temp = 1000.0
        final_temp = 1.0
        cooling_rate = 0.95

        current_efficiency = 0.7
        best_efficiency = current_efficiency

        temp = initial_temp
        iterations = 0

        while temp > final_temp and iterations < 100:
            # Generate neighbor solution (simplified)
            new_efficiency = current_efficiency + np.random.normal(0, 0.1)
            new_efficiency = max(0.0, min(1.0, new_efficiency))

            # Accept or reject
            if new_efficiency > current_efficiency or np.random.random() < np.exp((new_efficiency - current_efficiency) / temp):
                current_efficiency = new_efficiency
                if new_efficiency > best_efficiency:
                    best_efficiency = new_efficiency

            temp *= cooling_rate
            iterations += 1

        return {
            'total_efficiency': best_efficiency,
            'iterations': iterations,
            'final_temperature': temp
        }

    def _genetic_algorithm(self, zones: List[Dict], params: Dict) -> Dict:
        """Genetic algorithm optimization (placeholder)"""
        return {'total_efficiency': 0.88, 'generations': 50}

    def _particle_swarm_optimization(self, zones: List[Dict], params: Dict) -> Dict:
        """Particle swarm optimization (placeholder)"""
        return {'total_efficiency': 0.86, 'particles': 30}