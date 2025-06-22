import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, LineString
import networkx as nx
import json
from datetime import datetime

@dataclass
class FloorPlan:
    """Represents a single floor in a multi-floor building"""
    floor_id: str
    floor_number: int
    elevation: float
    floor_height: float
    zones: List[Dict]
    vertical_connections: List[Dict]  # stairs, elevators, etc.
    mechanical_spaces: List[Dict]
    structural_elements: List[Dict]
    analysis_results: Dict

@dataclass
class VerticalConnection:
    """Represents vertical circulation elements"""
    connection_id: str
    connection_type: str  # 'stair', 'elevator', 'escalator', 'ramp'
    served_floors: List[int]
    location: Tuple[float, float]
    capacity: int
    accessibility_compliant: bool
    fire_rated: bool

@dataclass
class BuildingCore:
    """Central building services and circulation core"""
    core_id: str
    location: Polygon
    services: List[str]  # elevators, stairs, restrooms, mechanical
    serves_floors: List[int]
    fire_rating: str
    pressurization: bool

class MultiFloorAnalyzer:
    """
    Advanced multi-floor building analysis for complex architectural projects
    """
    
    def __init__(self):
        self.building_floors = {}
        self.vertical_circulation = []
        self.building_cores = []
        self.structural_grid = {}
        self.mechanical_systems = {}
        self.fire_safety_systems = {}
        
        # Building code requirements
        self.code_requirements = {
            'max_travel_distance': {
                'office': 45,  # meters to exit
                'assembly': 30,
                'residential': 35
            },
            'min_egress_width': {
                'corridor': 1.2,  # meters
                'stair': 1.1,
                'door': 0.8
            },
            'elevator_requirements': {
                'max_floors_without_elevator': 3,
                'min_cars_per_population': 1/150  # 1 car per 150 people
            },
            'accessibility': {
                'accessible_route_required': True,
                'elevator_required_floors': 4,  # 4+ floors need elevator
                'accessible_parking_percentage': 0.02  # 2% of total
            }
        }
    
    def analyze_multi_floor_building(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Comprehensive analysis of multi-floor building"""
        
        # Sort floors by elevation
        floor_plans.sort(key=lambda f: f.elevation)
        self.building_floors = {f.floor_number: f for f in floor_plans}
        
        analysis_results = {
            'building_overview': self._analyze_building_overview(floor_plans),
            'vertical_circulation': self._analyze_vertical_circulation(floor_plans),
            'structural_analysis': self._analyze_structural_consistency(floor_plans),
            'mechanical_systems': self._analyze_mechanical_systems(floor_plans),
            'fire_safety': self._analyze_fire_safety(floor_plans),
            'accessibility': self._analyze_accessibility(floor_plans),
            'space_programming': self._analyze_space_programming(floor_plans),
            'efficiency_metrics': self._calculate_building_efficiency(floor_plans),
            'code_compliance': self._check_code_compliance(floor_plans),
            'optimization_recommendations': self._generate_optimization_recommendations(floor_plans)
        }
        
        return analysis_results
    
    def _analyze_building_overview(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Analyze overall building characteristics"""
        total_floors = len(floor_plans)
        total_area = sum(self._calculate_floor_area(floor.zones) for floor in floor_plans)
        building_height = max(f.elevation + f.floor_height for f in floor_plans) - min(f.elevation for f in floor_plans)
        
        # Calculate floor area ratios
        floor_areas = [self._calculate_floor_area(floor.zones) for floor in floor_plans]
        typical_floor_area = np.mean(floor_areas)
        area_variation = np.std(floor_areas) / typical_floor_area if typical_floor_area > 0 else 0
        
        # Identify building type based on characteristics
        building_type = self._classify_building_type(floor_plans)
        
        # Calculate occupancy
        total_occupancy = sum(self._calculate_floor_occupancy(floor) for floor in floor_plans)
        
        return {
            'total_floors': total_floors,
            'building_height': building_height,
            'total_gross_area': total_area,
            'typical_floor_area': typical_floor_area,
            'area_variation_coefficient': area_variation,
            'building_type': building_type,
            'total_occupancy': total_occupancy,
            'occupancy_per_floor': total_occupancy / total_floors if total_floors > 0 else 0,
            'floor_area_ratio': total_area / floor_areas[0] if floor_areas else 0,
            'building_efficiency': self._calculate_core_to_floor_ratio(floor_plans)
        }
    
    def _analyze_vertical_circulation(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Analyze vertical circulation systems"""
        
        # Find all vertical circulation elements
        stairs = []
        elevators = []
        other_circulation = []
        
        for floor in floor_plans:
            for connection in floor.vertical_connections:
                if connection['type'] == 'stair':
                    stairs.append(connection)
                elif connection['type'] == 'elevator':
                    elevators.append(connection)
                else:
                    other_circulation.append(connection)
        
        # Analyze stair distribution
        stair_analysis = self._analyze_stairs(stairs, floor_plans)
        
        # Analyze elevator system
        elevator_analysis = self._analyze_elevators(elevators, floor_plans)
        
        # Calculate circulation efficiency
        circulation_efficiency = self._calculate_circulation_efficiency(floor_plans)
        
        return {
            'total_stairs': len(stairs),
            'total_elevators': len(elevators),
            'stair_analysis': stair_analysis,
            'elevator_analysis': elevator_analysis,
            'circulation_efficiency': circulation_efficiency,
            'emergency_egress_compliance': self._check_egress_compliance(floor_plans),
            'accessibility_compliance': self._check_vertical_accessibility(floor_plans)
        }
    
    def _analyze_stairs(self, stairs: List[Dict], floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Detailed stair analysis"""
        if not stairs:
            return {'count': 0, 'compliance': 'No stairs found'}
        
        # Group stairs by location (same stair serving multiple floors)
        stair_groups = self._group_stairs_by_location(stairs)
        
        stair_analysis = {
            'stair_count': len(stair_groups),
            'total_stair_instances': len(stairs),
            'average_width': np.mean([s.get('width', 1.2) for s in stairs]),
            'fire_rated_count': sum(1 for s in stairs if s.get('fire_rated', False)),
            'pressurized_count': sum(1 for s in stairs if s.get('pressurized', False)),
            'compliance_issues': []
        }
        
        # Check code compliance
        min_width = self.code_requirements['min_egress_width']['stair']
        for stair in stairs:
            width = stair.get('width', 1.2)
            if width < min_width:
                stair_analysis['compliance_issues'].append(
                    f"Stair width {width}m below minimum {min_width}m"
                )
        
        return stair_analysis
    
    def _analyze_elevators(self, elevators: List[Dict], floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Detailed elevator analysis"""
        if not elevators:
            return {'count': 0, 'compliance': 'No elevators found'}
        
        total_occupancy = sum(self._calculate_floor_occupancy(floor) for floor in floor_plans)
        required_elevators = max(1, int(total_occupancy * self.code_requirements['elevator_requirements']['min_cars_per_population']))
        
        elevator_groups = self._group_elevators_by_bank(elevators)
        
        return {
            'elevator_count': len(set(e.get('elevator_id', i) for i, e in enumerate(elevators))),
            'elevator_banks': len(elevator_groups),
            'total_capacity': sum(e.get('capacity', 1000) for e in elevators),
            'average_capacity': np.mean([e.get('capacity', 1000) for e in elevators]),
            'required_elevators': required_elevators,
            'compliance': 'Adequate' if len(elevator_groups) >= required_elevators else 'Insufficient',
            'service_quality': self._calculate_elevator_service_quality(elevators, total_occupancy)
        }
    
    def _analyze_structural_consistency(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Analyze structural grid consistency across floors"""
        
        structural_analysis = {
            'grid_consistency': True,
            'typical_bay_size': None,
            'structural_variations': [],
            'column_alignment': {},
            'load_path_continuity': True
        }
        
        # Extract structural grids from each floor
        floor_grids = {}
        for floor in floor_plans:
            grid = self._extract_structural_grid(floor)
            floor_grids[floor.floor_number] = grid
        
        # Analyze grid consistency
        if len(floor_grids) > 1:
            reference_grid = list(floor_grids.values())[0]
            
            for floor_num, grid in floor_grids.items():
                consistency_score = self._compare_structural_grids(reference_grid, grid)
                if consistency_score < 0.8:  # 80% similarity threshold
                    structural_analysis['grid_consistency'] = False
                    structural_analysis['structural_variations'].append({
                        'floor': floor_num,
                        'consistency_score': consistency_score,
                        'issues': 'Significant deviation from typical grid'
                    })
        
        # Calculate typical bay size
        if floor_grids:
            bay_sizes = []
            for grid in floor_grids.values():
                bay_sizes.extend(self._calculate_bay_sizes(grid))
            
            if bay_sizes:
                structural_analysis['typical_bay_size'] = {
                    'average': np.mean(bay_sizes),
                    'standard_deviation': np.std(bay_sizes),
                    'range': [min(bay_sizes), max(bay_sizes)]
                }
        
        return structural_analysis
    
    def _analyze_mechanical_systems(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Analyze mechanical, electrical, and plumbing systems"""
        
        mechanical_analysis = {
            'hvac_zones': [],
            'electrical_distribution': {},
            'plumbing_risers': [],
            'system_efficiency': {},
            'equipment_locations': {},
            'service_accessibility': {}
        }
        
        # Analyze HVAC systems
        hvac_zones = []
        for floor in floor_plans:
            floor_hvac = self._analyze_floor_hvac(floor)
            hvac_zones.extend(floor_hvac)
        
        mechanical_analysis['hvac_zones'] = self._optimize_hvac_zoning(hvac_zones)
        
        # Analyze electrical systems
        mechanical_analysis['electrical_distribution'] = self._analyze_electrical_distribution(floor_plans)
        
        # Analyze plumbing systems
        mechanical_analysis['plumbing_risers'] = self._analyze_plumbing_systems(floor_plans)
        
        # Calculate system efficiency
        mechanical_analysis['system_efficiency'] = self._calculate_mep_efficiency(floor_plans)
        
        return mechanical_analysis
    
    def _analyze_fire_safety(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Comprehensive fire safety analysis"""
        
        fire_safety_analysis = {
            'egress_analysis': {},
            'compartmentalization': {},
            'fire_rated_assemblies': [],
            'suppression_systems': {},
            'alarm_systems': {},
            'emergency_lighting': {},
            'compliance_score': 0.0
        }
        
        # Analyze egress for each floor
        egress_compliance = []
        for floor in floor_plans:
            floor_egress = self._analyze_floor_egress(floor)
            egress_compliance.append(floor_egress)
        
        fire_safety_analysis['egress_analysis'] = {
            'floors_compliant': sum(1 for e in egress_compliance if e['compliant']),
            'total_floors': len(floor_plans),
            'max_travel_distance': max(e['max_travel_distance'] for e in egress_compliance),
            'egress_width_adequate': all(e['width_adequate'] for e in egress_compliance)
        }
        
        # Analyze fire compartmentalization
        fire_safety_analysis['compartmentalization'] = self._analyze_fire_compartments(floor_plans)
        
        # Calculate compliance score
        compliance_factors = [
            fire_safety_analysis['egress_analysis']['floors_compliant'] / len(floor_plans),
            1.0 if fire_safety_analysis['egress_analysis']['egress_width_adequate'] else 0.5,
            fire_safety_analysis['compartmentalization'].get('compliance_score', 0.5)
        ]
        
        fire_safety_analysis['compliance_score'] = np.mean(compliance_factors)
        
        return fire_safety_analysis
    
    def _analyze_accessibility(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Analyze ADA and accessibility compliance"""
        
        accessibility_analysis = {
            'accessible_routes': {},
            'elevator_access': {},
            'accessible_spaces': {},
            'parking_compliance': {},
            'toilet_facilities': {},
            'compliance_level': 'Unknown'
        }
        
        # Check elevator requirements
        total_floors = len(floor_plans)
        elevators_required = total_floors >= self.code_requirements['accessibility']['elevator_required_floors']
        
        elevators_present = any(
            any(conn['type'] == 'elevator' for conn in floor.vertical_connections)
            for floor in floor_plans
        )
        
        accessibility_analysis['elevator_access'] = {
            'required': elevators_required,
            'provided': elevators_present,
            'compliant': not elevators_required or elevators_present
        }
        
        # Analyze accessible routes on each floor
        accessible_routes = []
        for floor in floor_plans:
            route_analysis = self._analyze_accessible_routes(floor)
            accessible_routes.append(route_analysis)
        
        accessibility_analysis['accessible_routes'] = {
            'floors_with_accessible_routes': sum(1 for r in accessible_routes if r['has_accessible_route']),
            'total_floors': len(floor_plans),
            'route_width_compliant': all(r['width_compliant'] for r in accessible_routes)
        }
        
        # Determine overall compliance level
        compliance_score = (
            (1.0 if accessibility_analysis['elevator_access']['compliant'] else 0.0) +
            (accessibility_analysis['accessible_routes']['floors_with_accessible_routes'] / total_floors)
        ) / 2
        
        if compliance_score >= 0.9:
            accessibility_analysis['compliance_level'] = 'Fully Compliant'
        elif compliance_score >= 0.7:
            accessibility_analysis['compliance_level'] = 'Mostly Compliant'
        elif compliance_score >= 0.5:
            accessibility_analysis['compliance_level'] = 'Partially Compliant'
        else:
            accessibility_analysis['compliance_level'] = 'Non-Compliant'
        
        return accessibility_analysis
    
    def _analyze_space_programming(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Analyze space programming and utilization across floors"""
        
        # Aggregate space types across all floors
        space_inventory = {}
        total_area_by_type = {}
        
        for floor in floor_plans:
            floor_spaces = self._extract_floor_spaces(floor)
            
            for space_type, spaces in floor_spaces.items():
                if space_type not in space_inventory:
                    space_inventory[space_type] = []
                    total_area_by_type[space_type] = 0
                
                space_inventory[space_type].extend(spaces)
                total_area_by_type[space_type] += sum(s['area'] for s in spaces)
        
        # Calculate programming metrics
        total_building_area = sum(total_area_by_type.values())
        
        programming_analysis = {
            'space_types': list(space_inventory.keys()),
            'space_distribution': {
                space_type: {
                    'count': len(spaces),
                    'total_area': total_area_by_type[space_type],
                    'percentage': (total_area_by_type[space_type] / total_building_area * 100) if total_building_area > 0 else 0,
                    'average_size': np.mean([s['area'] for s in spaces]) if spaces else 0
                }
                for space_type, spaces in space_inventory.items()
            },
            'programming_efficiency': self._calculate_programming_efficiency(space_inventory),
            'space_utilization': self._analyze_space_utilization(floor_plans),
            'vertical_stacking': self._analyze_vertical_stacking(floor_plans)
        }
        
        return programming_analysis
    
    def _calculate_building_efficiency(self, floor_plans: List[FloorPlan]) -> Dict[str, Any]:
        """Calculate overall building efficiency metrics"""
        
        # Calculate various efficiency ratios
        total_gross_area = sum(self._calculate_floor_area(floor.zones) for floor in floor_plans)
        total_net_area = sum(self._calculate_net_floor_area(floor.zones) for floor in floor_plans)
        total_circulation_area = sum(self._calculate_circulation_area(floor) for floor in floor_plans)
        total_core_area = sum(self._calculate_core_area(floor) for floor in floor_plans)
        
        efficiency_metrics = {
            'net_to_gross_ratio': (total_net_area / total_gross_area) if total_gross_area > 0 else 0,
            'circulation_efficiency': (total_circulation_area / total_gross_area) if total_gross_area > 0 else 0,
            'core_efficiency': (total_core_area / total_gross_area) if total_gross_area > 0 else 0,
            'usable_area_ratio': ((total_net_area - total_circulation_area) / total_gross_area) if total_gross_area > 0 else 0,
            'floor_plate_efficiency': self._calculate_floor_plate_efficiency(floor_plans),
            'vertical_efficiency': self._calculate_vertical_efficiency(floor_plans),
            'overall_efficiency_score': 0.0
        }
        
        # Calculate overall efficiency score
        efficiency_scores = [
            efficiency_metrics['net_to_gross_ratio'],
            1.0 - efficiency_metrics['circulation_efficiency'],  # Lower circulation is better
            1.0 - efficiency_metrics['core_efficiency'],  # Lower core percentage is better
            efficiency_metrics['usable_area_ratio'],
            efficiency_metrics['floor_plate_efficiency'],
            efficiency_metrics['vertical_efficiency']
        ]
        
        efficiency_metrics['overall_efficiency_score'] = np.mean([s for s in efficiency_scores if s > 0])
        
        return efficiency_metrics
    
    def _generate_optimization_recommendations(self, floor_plans: List[FloorPlan]) -> List[Dict[str, Any]]:
        """Generate specific recommendations for building optimization"""
        
        recommendations = []
        
        # Analyze circulation efficiency
        circulation_analysis = self._analyze_vertical_circulation(floor_plans)
        if circulation_analysis['circulation_efficiency'] < 0.15:  # Less than 15% is good
            recommendations.append({
                'category': 'Circulation',
                'priority': 'Medium',
                'recommendation': 'Consider optimizing vertical circulation layout to reduce circulation area',
                'potential_benefit': 'Increase usable area by 3-5%'
            })
        
        # Analyze space utilization
        space_analysis = self._analyze_space_programming(floor_plans)
        underutilized_spaces = [
            space_type for space_type, data in space_analysis['space_distribution'].items()
            if data['average_size'] < 10 and data['count'] > 5
        ]
        
        if underutilized_spaces:
            recommendations.append({
                'category': 'Space Planning',
                'priority': 'High',
                'recommendation': f'Consider consolidating small {", ".join(underutilized_spaces)} spaces',
                'potential_benefit': 'Improve space efficiency and reduce operational costs'
            })
        
        # Analyze structural efficiency
        structural_analysis = self._analyze_structural_consistency(floor_plans)
        if not structural_analysis['grid_consistency']:
            recommendations.append({
                'category': 'Structural',
                'priority': 'High',
                'recommendation': 'Standardize structural grid across floors to reduce construction costs',
                'potential_benefit': 'Reduce construction cost by 5-10%'
            })
        
        # Analyze mechanical systems
        mechanical_analysis = self._analyze_mechanical_systems(floor_plans)
        if mechanical_analysis['system_efficiency'].get('hvac_efficiency', 0.5) < 0.7:
            recommendations.append({
                'category': 'MEP Systems',
                'priority': 'Medium',
                'recommendation': 'Optimize HVAC zoning and equipment placement',
                'potential_benefit': 'Reduce energy consumption by 15-20%'
            })
        
        # Add fire safety recommendations
        fire_analysis = self._analyze_fire_safety(floor_plans)
        if fire_analysis['compliance_score'] < 0.8:
            recommendations.append({
                'category': 'Fire Safety',
                'priority': 'Critical',
                'recommendation': 'Address fire safety compliance issues identified in analysis',
                'potential_benefit': 'Ensure code compliance and occupant safety'
            })
        
        return sorted(recommendations, key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}[x['priority']])
    
    # Helper methods for detailed analysis
    
    def _calculate_floor_area(self, zones: List[Dict]) -> float:
        """Calculate total floor area"""
        total_area = 0
        for zone in zones:
            if zone.get('points'):
                poly = Polygon(zone['points'])
                total_area += poly.area
        return total_area
    
    def _calculate_floor_occupancy(self, floor: FloorPlan) -> int:
        """Calculate floor occupancy based on space types"""
        total_occupancy = 0
        
        if hasattr(floor, 'analysis_results') and floor.analysis_results:
            rooms = floor.analysis_results.get('rooms', {})
            
            occupancy_rates = {
                'Office': 10,  # m² per person
                'Open Office': 8,
                'Conference Room': 2,
                'Meeting Room': 3,
                'Reception': 15,
                'Break Room': 4
            }
            
            for zone_name, room_info in rooms.items():
                space_type = room_info.get('type', 'Office')
                area = room_info.get('area', 0)
                rate = occupancy_rates.get(space_type, 12)
                
                if rate > 0:
                    total_occupancy += max(1, int(area / rate))
        
        return total_occupancy
    
    def _classify_building_type(self, floor_plans: List[FloorPlan]) -> str:
        """Classify building type based on space programming"""
        
        # Aggregate space types across all floors
        space_areas = {}
        total_area = 0
        
        for floor in floor_plans:
            if hasattr(floor, 'analysis_results') and floor.analysis_results:
                rooms = floor.analysis_results.get('rooms', {})
                
                for room_info in rooms.values():
                    space_type = room_info.get('type', 'Unknown')
                    area = room_info.get('area', 0)
                    
                    if space_type not in space_areas:
                        space_areas[space_type] = 0
                    space_areas[space_type] += area
                    total_area += area
        
        if not space_areas or total_area == 0:
            return 'Unknown'
        
        # Calculate percentages
        space_percentages = {space_type: (area / total_area) 
                           for space_type, area in space_areas.items()}
        
        # Classify based on dominant space types
        office_percentage = space_percentages.get('Office', 0) + space_percentages.get('Open Office', 0)
        
        if office_percentage > 0.6:
            return 'Office Building'
        elif space_percentages.get('Conference Room', 0) + space_percentages.get('Meeting Room', 0) > 0.3:
            return 'Corporate Headquarters'
        elif space_percentages.get('Reception', 0) > 0.2:
            return 'Commercial Office'
        else:
            return 'Mixed Use'
    
    def _extract_structural_grid(self, floor: FloorPlan) -> Dict:
        """Extract structural grid from floor plan"""
        # This is a simplified implementation
        # In practice, would analyze structural elements from the floor plan
        
        grid = {
            'columns': [],
            'beams': [],
            'grid_lines': [],
            'bay_sizes': []
        }
        
        # Extract from structural_elements if available
        if hasattr(floor, 'structural_elements'):
            for element in floor.structural_elements:
                if element.get('type') == 'column':
                    grid['columns'].append(element)
                elif element.get('type') == 'beam':
                    grid['beams'].append(element)
        
        return grid
    
    def _compare_structural_grids(self, grid1: Dict, grid2: Dict) -> float:
        """Compare two structural grids for consistency"""
        # Simplified comparison - in practice would compare column locations, beam spans, etc.
        
        if not grid1 or not grid2:
            return 0.5
        
        # Compare number of structural elements
        col_similarity = 1.0 - abs(len(grid1.get('columns', [])) - len(grid2.get('columns', []))) / max(len(grid1.get('columns', [])), len(grid2.get('columns', [])), 1)
        beam_similarity = 1.0 - abs(len(grid1.get('beams', [])) - len(grid2.get('beams', []))) / max(len(grid1.get('beams', [])), len(grid2.get('beams', [])), 1)
        
        return (col_similarity + beam_similarity) / 2
    
    def _calculate_bay_sizes(self, grid: Dict) -> List[float]:
        """Calculate structural bay sizes from grid"""
        # Simplified implementation
        return [6.0, 7.5, 6.0, 7.5]  # Default bay sizes
    
    def _analyze_floor_hvac(self, floor: FloorPlan) -> List[Dict]:
        """Analyze HVAC requirements for a floor"""
        hvac_zones = []
        
        if hasattr(floor, 'analysis_results') and floor.analysis_results:
            rooms = floor.analysis_results.get('rooms', {})
            
            for zone_name, room_info in rooms.items():
                hvac_zone = {
                    'zone_id': f"{floor.floor_number}_{zone_name}",
                    'space_type': room_info.get('type', 'Office'),
                    'area': room_info.get('area', 0),
                    'cooling_load': self._calculate_cooling_load(room_info),
                    'heating_load': self._calculate_heating_load(room_info),
                    'ventilation_requirement': self._calculate_ventilation_requirement(room_info)
                }
                hvac_zones.append(hvac_zone)
        
        return hvac_zones
    
    def _calculate_cooling_load(self, room_info: Dict) -> float:
        """Calculate cooling load for a space"""
        area = room_info.get('area', 0)
        space_type = room_info.get('type', 'Office')
        
        # Simplified cooling load calculation (W/m²)
        load_factors = {
            'Office': 150,
            'Conference Room': 200,
            'Open Office': 130,
            'Reception': 120,
            'Kitchen': 300,
            'Server Room': 500
        }
        
        load_factor = load_factors.get(space_type, 150)
        return area * load_factor
    
    def _calculate_heating_load(self, room_info: Dict) -> float:
        """Calculate heating load for a space"""
        # Simplified - usually 60-70% of cooling load
        return self._calculate_cooling_load(room_info) * 0.65
    
    def _calculate_ventilation_requirement(self, room_info: Dict) -> float:
        """Calculate ventilation requirement (L/s)"""
        area = room_info.get('area', 0)
        space_type = room_info.get('type', 'Office')
        
        # Ventilation rates (L/s/m²)
        ventilation_rates = {
            'Office': 2.5,
            'Conference Room': 5.0,
            'Open Office': 2.5,
            'Reception': 2.0,
            'Kitchen': 10.0,
            'Bathroom': 25.0
        }
        
        rate = ventilation_rates.get(space_type, 2.5)
        return area * rate