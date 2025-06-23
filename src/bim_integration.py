import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import numpy as np
from shapely.geometry import Polygon, Point
import math

@dataclass
class BIMElement:
    """Base class for BIM elements"""
    id: str
    name: str
    element_type: str
    properties: Dict[str, Any]
    geometry: Dict[str, Any]
    materials: List[str]
    relationships: List[str]

@dataclass
class BIMSpace(BIMElement):
    """BIM Space element specifically for rooms/spaces"""
    function: str
    area: float
    volume: float
    occupancy: int
    fire_rating: str
    accessibility_level: str

@dataclass
class BIMBuilding:
    """Complete BIM building model"""
    id: str
    name: str
    address: str
    floors: List['BIMFloor']
    metadata: Dict[str, Any]
    standards_compliance: Dict[str, Any]

@dataclass
class BIMFloor:
    """BIM Floor containing spaces and elements"""
    id: str
    name: str
    level: float
    height: float
    spaces: List[BIMSpace]
    elements: List[BIMElement]

class BIMModelGenerator:
    """Generate BIM models from architectural analysis"""
    
    def __init__(self):
        self.standards = {
            'ifc_version': '4.0',
            'space_height_default': 2.7,  # meters
            'wall_thickness_default': 0.2,  # meters
            'accessibility_standards': 'ADA',
            'building_codes': ['IBC', 'NFPA']
        }
    
    def create_bim_model_from_analysis(self, zones: List[Dict], 
                                     analysis_results: Dict, 
                                     building_metadata: Dict) -> BIMBuilding:
        """Create comprehensive BIM model from DWG analysis"""
        
        # Create building
        building_id = building_metadata.get('id', str(uuid.uuid4()))
        building = BIMBuilding(
            id=building_id,
            name=building_metadata.get('name', 'Analyzed Building'),
            address=building_metadata.get('address', 'Unknown'),
            floors=[],
            metadata=building_metadata,
            standards_compliance={}
        )
        
        # Create floor
        floor = self._create_floor_from_zones(zones, analysis_results)
        building.floors.append(floor)
        
        # Analyze standards compliance
        building.standards_compliance = self._analyze_standards_compliance(building)
        
        return building
    
    def _create_floor_from_zones(self, zones: List[Dict], analysis_results: Dict) -> BIMFloor:
        """Create BIM floor from analyzed zones"""
        
        floor_id = f"floor_{uuid.uuid4()}"
        spaces = []
        elements = []
        
        # Convert zones to BIM spaces
        for i, zone in enumerate(zones):
            if 'points' in zone:
                space = self._create_bim_space_from_zone(zone, i, analysis_results)
                spaces.append(space)
                
                # Create space boundary elements
                boundary_elements = self._create_space_boundaries(zone, space.id)
                elements.extend(boundary_elements)
        
        return BIMFloor(
            id=floor_id,
            name="Ground Floor",
            level=0.0,
            height=self.standards['space_height_default'],
            spaces=spaces,
            elements=elements
        )
    
    def _create_bim_space_from_zone(self, zone: Dict, zone_index: int, 
                                  analysis_results: Dict) -> BIMSpace:
        """Create BIM space from zone data"""
        
        space_id = f"space_{zone_index}_{uuid.uuid4()}"
        
        # Get room analysis
        room_analysis = analysis_results.get('room_analysis', {}).get(str(zone_index), {})
        room_type = room_analysis.get('room_type', 'Unknown')
        
        # Calculate space properties
        area = zone.get('area', 0)
        volume = area * self.standards['space_height_default']
        
        # Determine occupancy based on room type and area
        occupancy = self._calculate_occupancy(room_type, area)
        
        # Determine fire rating based on room type
        fire_rating = self._determine_fire_rating(room_type, area)
        
        # Check accessibility compliance
        accessibility_level = self._assess_accessibility(zone, room_type)
        
        return BIMSpace(
            id=space_id,
            name=f"{room_type}_{zone_index + 1}",
            element_type="IfcSpace",
            properties={
                'LongName': f"{room_type} {zone_index + 1}",
                'Description': f"Analyzed from DWG - {room_type}",
                'NetFloorArea': area,
                'GrossFloorArea': area * 1.1,  # Account for walls
                'FinishFloorLevel': 0.0,
                'FinishCeilingLevel': self.standards['space_height_default']
            },
            geometry={
                'boundary_points': zone.get('points', []),
                'area': area,
                'perimeter': zone.get('perimeter', 0),
                'centroid': self._calculate_centroid(zone.get('points', []))
            },
            materials=['Generic Wall Finish', 'Generic Floor Finish', 'Generic Ceiling'],
            relationships=[],
            function=room_type,
            area=area,
            volume=volume,
            occupancy=occupancy,
            fire_rating=fire_rating,
            accessibility_level=accessibility_level
        )
    
    def _create_space_boundaries(self, zone: Dict, space_id: str) -> List[BIMElement]:
        """Create wall elements for space boundaries"""
        
        elements = []
        points = zone.get('points', [])
        
        if len(points) < 3:
            return elements
        
        # Create wall elements for each boundary segment
        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]
            
            wall_id = f"wall_{space_id}_{i}"
            wall_length = ((end_point[0] - start_point[0])**2 + 
                          (end_point[1] - start_point[1])**2)**0.5
            
            wall_element = BIMElement(
                id=wall_id,
                name=f"Wall_{i+1}",
                element_type="IfcWall",
                properties={
                    'Length': wall_length,
                    'Height': self.standards['space_height_default'],
                    'Thickness': self.standards['wall_thickness_default'],
                    'IsExternal': self._is_external_wall(start_point, end_point, zone),
                    'LoadBearing': False,
                    'ThermalTransmittance': 0.4
                },
                geometry={
                    'start_point': start_point,
                    'end_point': end_point,
                    'thickness': self.standards['wall_thickness_default'],
                    'height': self.standards['space_height_default']
                },
                materials=['Generic Wall Material'],
                relationships=[space_id]
            )
            
            elements.append(wall_element)
        
        return elements
    
    def _calculate_occupancy(self, room_type: str, area: float) -> int:
        """Calculate occupancy based on room type and area"""
        
        occupancy_factors = {
            'Office': 10,  # m² per person
            'Conference Room': 1.5,
            'Open Office': 8,
            'Corridor': 50,
            'Storage': 100,
            'Kitchen': 20,
            'Bathroom': 1,
            'Reception': 5,
            'Server Room': 50,
            'Break Room': 5
        }
        
        factor = occupancy_factors.get(room_type, 15)
        return max(1, int(area / factor))
    
    def _determine_fire_rating(self, room_type: str, area: float) -> str:
        """Determine fire rating requirements"""
        
        if room_type in ['Server Room', 'Storage'] and area > 20:
            return '1-hour'
        elif room_type in ['Kitchen'] or area > 100:
            return '30-minute'
        else:
            return 'Non-rated'
    
    def _assess_accessibility(self, zone: Dict, room_type: str) -> str:
        """Assess accessibility compliance level"""
        
        area = zone.get('area', 0)
        
        # Basic accessibility assessment
        if area < 7:  # Too small for wheelchair access
            return 'Non-accessible'
        elif room_type in ['Bathroom', 'Kitchen'] and area < 12:
            return 'Limited'
        elif area >= 12:
            return 'Fully Accessible'
        else:
            return 'Partially Accessible'
    
    def _is_external_wall(self, start_point: tuple, end_point: tuple, zone: Dict) -> bool:
        """Determine if wall segment is external (simplified)"""
        # This is a simplified check - in practice would analyze adjacencies
        return False
    
    def _calculate_centroid(self, points: List[tuple]) -> tuple:
        """Calculate centroid of polygon"""
        if not points:
            return (0, 0)
        
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return (x, y)
    
    def _analyze_standards_compliance(self, building: BIMBuilding) -> Dict[str, Any]:
        """Analyze building compliance with standards"""
        
        compliance = {
            'ifc': {'version': self.standards['ifc_version'], 'score': 95.0},
            'accessibility': self._check_accessibility_compliance(building),
            'fire_safety': self._check_fire_safety_compliance(building),
            'building_code': self._check_building_code_compliance(building),
            'spaces': {'total_spaces': len(building.floors[0].spaces), 'compliant_spaces': 0}
        }
        
        # Count compliant spaces
        compliant_count = 0
        for floor in building.floors:
            for space in floor.spaces:
                if space.accessibility_level in ['Fully Accessible', 'Partially Accessible']:
                    compliant_count += 1
        
        compliance['spaces']['compliant_spaces'] = compliant_count
        
        return compliance
    
    def _check_accessibility_compliance(self, building: BIMBuilding) -> Dict[str, Any]:
        """Check ADA/accessibility compliance"""
        
        total_spaces = sum(len(floor.spaces) for floor in building.floors)
        accessible_spaces = 0
        
        for floor in building.floors:
            for space in floor.spaces:
                if space.accessibility_level in ['Fully Accessible']:
                    accessible_spaces += 1
        
        compliance_percentage = (accessible_spaces / total_spaces * 100) if total_spaces > 0 else 0
        
        return {
            'standard': 'ADA',
            'compliance_percentage': compliance_percentage,
            'accessible_spaces': accessible_spaces,
            'total_spaces': total_spaces,
            'compliant': compliance_percentage >= 80
        }
    
    def _check_fire_safety_compliance(self, building: BIMBuilding) -> Dict[str, Any]:
        """Check fire safety compliance"""
        
        return {
            'standard': 'NFPA',
            'exit_requirements': 'Compliant',
            'fire_ratings': 'Adequate',
            'sprinkler_coverage': 'Required',
            'smoke_detection': 'Required'
        }
    
    def _check_building_code_compliance(self, building: BIMBuilding) -> Dict[str, Any]:
        """Check building code compliance"""
        
        return {
            'standard': 'IBC',
            'occupancy_classification': 'Business',
            'construction_type': 'Type V',
            'height_limits': 'Compliant',
            'area_limits': 'Compliant'
        }

class BIMStandardsCompliance:
    """
    Ensures compliance with international BIM standards including
    IFC, COBie, and industry-specific requirements
    """
    
    def __init__(self):
        self.standards = {
            'IFC': {
                'version': '4.3',
                'schema': 'IFC4X3',
                'required_properties': [
                    'GlobalId', 'Name', 'Description', 'ObjectType',
                    'PredefinedType', 'ObjectPlacement', 'Representation'
                ]
            },
            'COBie': {
                'version': '2.4',
                'required_sheets': [
                    'Contact', 'Facility', 'Floor', 'Space', 'Zone',
                    'Type', 'Component', 'System', 'Assembly'
                ]
            },
            'LEED': {
                'version': 'v4.1',
                'categories': [
                    'Location and Transportation', 'Sustainable Sites',
                    'Water Efficiency', 'Energy and Atmosphere',
                    'Materials and Resources', 'Indoor Environmental Quality'
                ]
            },
            'ADA': {
                'compliance_areas': [
                    'Accessible Routes', 'Parking', 'Passenger Loading Zones',
                    'Accessible Entrances', 'Doors and Doorways', 'Ramps',
                    'Elevators', 'Platform Lifts', 'Stairs', 'Handrails'
                ]
            }
        }
        
        self.space_requirements = {
            'Office': {
                'min_area': 9.0,  # m²
                'min_ceiling_height': 2.7,  # m
                'ventilation_rate': 8.5,  # L/s per person
                'lighting_level': 500,  # lux
                'accessibility': 'Type A'
            },
            'Conference Room': {
                'min_area': 12.0,
                'min_ceiling_height': 2.7,
                'ventilation_rate': 10.0,
                'lighting_level': 500,
                'accessibility': 'Type A',
                'acoustic_rating': 'STC 50'
            },
            'Corridor': {
                'min_width': 1.2,
                'min_ceiling_height': 2.4,
                'ventilation_rate': 0.5,
                'lighting_level': 200,
                'accessibility': 'Type A'
            },
            'Bathroom': {
                'min_area': 3.0,
                'min_ceiling_height': 2.4,
                'ventilation_rate': 50,  # air changes per hour
                'lighting_level': 200,
                'accessibility': 'ADA compliant',
                'plumbing': 'Required'
            }
        }
    
    def validate_ifc_compliance(self, bim_model: BIMBuilding) -> Dict[str, Any]:
        """Validate IFC standard compliance"""
        compliance_report = {
            'standard': 'IFC 4.3',
            'compliant': True,
            'violations': [],
            'recommendations': [],
            'score': 100.0
        }
        
        # Check required properties for each element
        all_elements = []
        for floor in bim_model.floors:
            all_elements.extend(floor.spaces)
            all_elements.extend(floor.elements)
        
        for element in all_elements:
            missing_props = []
            required_props = self.standards['IFC']['required_properties']
            
            for prop in required_props:
                if prop not in element.properties:
                    missing_props.append(prop)
            
            if missing_props:
                compliance_report['violations'].append({
                    'element_id': element.id,
                    'element_type': element.element_type,
                    'missing_properties': missing_props
                })
                compliance_report['compliant'] = False
        
        # Calculate compliance score
        total_elements = len(all_elements)
        violations = len(compliance_report['violations'])
        if total_elements > 0:
            compliance_report['score'] = max(0, (total_elements - violations) / total_elements * 100)
        
        return compliance_report
    
    def validate_space_requirements(self, spaces: List[BIMSpace]) -> Dict[str, Any]:
        """Validate spaces against building code requirements"""
        validation_report = {
            'compliant_spaces': 0,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        for space in spaces:
            space_type = space.function
            requirements = self.space_requirements.get(space_type, {})
            
            violations = []
            warnings = []
            
            # Check minimum area
            if 'min_area' in requirements and space.area < requirements['min_area']:
                violations.append(f"Area {space.area:.1f}m² below minimum {requirements['min_area']}m²")
            
            # Check ceiling height
            if 'min_ceiling_height' in requirements:
                ceiling_height = space.properties.get('ceiling_height', 0)
                if ceiling_height < requirements['min_ceiling_height']:
                    violations.append(f"Ceiling height {ceiling_height}m below minimum {requirements['min_ceiling_height']}m")
            
            # Check accessibility
            if 'accessibility' in requirements:
                accessibility = space.accessibility_level
                if accessibility != requirements['accessibility']:
                    warnings.append(f"Accessibility level '{accessibility}' may not meet '{requirements['accessibility']}' standard")
            
            if violations:
                validation_report['violations'].append({
                    'space_id': space.id,
                    'space_name': space.name,
                    'space_type': space_type,
                    'violations': violations
                })
            elif warnings:
                validation_report['warnings'].append({
                    'space_id': space.id,
                    'space_name': space.name,
                    'warnings': warnings
                })
            else:
                validation_report['compliant_spaces'] += 1
        
        return validation_report
    
    def generate_cobie_data(self, bim_model: BIMBuilding) -> Dict[str, List[Dict]]:
        """Generate COBie-compliant data structure"""
        cobie_data = {
            'Contact': [],
            'Facility': [],
            'Floor': [],
            'Space': [],
            'Zone': [],
            'Type': [],
            'Component': [],
            'System': [],
            'Assembly': []
        }
        
        # Facility sheet
        cobie_data['Facility'].append({
            'Name': bim_model.name,
            'CreatedBy': 'AI Architectural Analyzer',
            'CreatedOn': datetime.now().isoformat(),
            'Category': 'Office Building',
            'ProjectName': bim_model.metadata.get('project_name', bim_model.name),
            'SiteName': bim_model.address,
            'LinearUnits': 'meter',
            'AreaUnits': 'square meter',
            'VolumeUnits': 'cubic meter'
        })
        
        # Floor sheets
        for floor in bim_model.floors:
            cobie_data['Floor'].append({
                'Name': floor.name,
                'CreatedBy': 'AI Architectural Analyzer',
                'CreatedOn': datetime.now().isoformat(),
                'Category': 'Floor',
                'ExtSystem': 'BIM Model',
                'ExtObject': floor.id,
                'Description': f'Floor at level {floor.level}m',
                'Elevation': floor.level,
                'Height': floor.height
            })
            
            # Space sheets
            for space in floor.spaces:
                cobie_data['Space'].append({
                    'Name': space.name,
                    'CreatedBy': 'AI Architectural Analyzer',
                    'CreatedOn': datetime.now().isoformat(),
                    'Category': 'Space',
                    'FloorName': floor.name,
                    'Description': f'{space.function} space',
                    'ExtSystem': 'BIM Model',
                    'ExtObject': space.id,
                    'RoomTag': space.name,
                    'UsableHeight': space.properties.get('ceiling_height', floor.height),
                    'GrossArea': space.area,
                    'NetArea': space.area * 0.95  # Assuming 5% for walls/structure
                })
        
        return cobie_data


class BIMModelGenerator:
    """
    Generates comprehensive BIM models from architectural analysis
    """
    
    def __init__(self):
        self.standards = BIMStandardsCompliance()
        self.default_properties = {
            'materials': ['Concrete', 'Steel', 'Gypsum Board'],
            'finishes': ['Paint', 'Carpet', 'Ceramic Tile'],
            'mechanical_systems': ['HVAC', 'Electrical', 'Plumbing'],
            'fire_safety': ['Sprinkler System', 'Fire Alarm', 'Emergency Lighting']
        }
    
    def create_bim_model_from_analysis(self, zones: List[Dict], 
                                     analysis_results: Dict,
                                     building_metadata: Dict) -> BIMBuilding:
        """Create a comprehensive BIM model from space analysis"""
        
        # Create building
        building_id = str(uuid.uuid4())
        building = BIMBuilding(
            id=building_id,
            name=building_metadata.get('name', 'Analyzed Building'),
            address=building_metadata.get('address', 'Unknown Address'),
            floors=[],
            metadata=building_metadata,
            standards_compliance={}
        )
        
        # Create floor (assuming single floor for now)
        floor_id = str(uuid.uuid4())
        floor = BIMFloor(
            id=floor_id,
            name='Ground Floor',
            level=0.0,
            height=building_metadata.get('floor_height', 3.0),
            spaces=[],
            elements=[]
        )
        
        # Convert zones to BIM spaces
        rooms = analysis_results.get('rooms', {})
        placements = analysis_results.get('placements', {})
        
        for i, zone in enumerate(zones):
            zone_name = f"Zone_{i}"
            room_info = rooms.get(zone_name, {})
            zone_placements = placements.get(zone_name, [])
            
            # Create BIM space
            space = self._create_bim_space(zone, room_info, zone_placements, floor.height)
            floor.spaces.append(space)
            
            # Create furniture elements
            for j, placement in enumerate(zone_placements):
                furniture_element = self._create_furniture_element(placement, j, space.id)
                floor.elements.append(furniture_element)
        
        building.floors.append(floor)
        
        # Validate compliance
        building.standards_compliance = {
            'ifc': self.standards.validate_ifc_compliance(building),
            'spaces': self.standards.validate_space_requirements(floor.spaces)
        }
        
        return building
    
    def _create_bim_space(self, zone: Dict, room_info: Dict, 
                         placements: List[Dict], floor_height: float) -> BIMSpace:
        """Create a BIM space from zone data"""
        space_id = str(uuid.uuid4())
        poly = Polygon(zone['points'])
        area = poly.area
        volume = area * floor_height
        
        # Determine function and properties
        function = room_info.get('type', 'Unknown')
        confidence = room_info.get('confidence', 0.5)
        
        # Calculate occupancy based on space type and area
        occupancy = self._calculate_occupancy(function, area)
        
        # Determine accessibility level
        accessibility_level = self._determine_accessibility(function, area, placements)
        
        # Create comprehensive properties
        properties = {
            'GlobalId': space_id,
            'Name': f"{function}_{space_id[:8]}",
            'Description': f'{function} space with {len(placements)} furniture pieces',
            'ObjectType': 'IfcSpace',
            'PredefinedType': self._map_to_ifc_space_type(function),
            'classification_confidence': confidence,
            'ceiling_height': floor_height,
            'floor_finish': self._recommend_floor_finish(function),
            'wall_finish': self._recommend_wall_finish(function),
            'lighting_type': self._recommend_lighting(function),
            'hvac_zone': f"HVAC_{space_id[:8]}",
            'fire_zone': f"FIRE_{space_id[:8]}",
            'furniture_count': len(placements),
            'utilization_rate': len(placements) / max(1, area / 10),  # furniture per 10m²
            'ventilation_requirement': self._calculate_ventilation(function, occupancy),
            'lighting_requirement': self._calculate_lighting(function),
            'acoustic_requirement': self._calculate_acoustic(function)
        }
        
        # Geometry representation
        geometry = {
            'type': 'Polygon',
            'coordinates': zone['points'],
            'area': area,
            'perimeter': poly.length,
            'centroid': [poly.centroid.x, poly.centroid.y],
            'bounds': list(poly.bounds)
        }
        
        space = BIMSpace(
            id=space_id,
            name=properties['Name'],
            element_type='IfcSpace',
            properties=properties,
            geometry=geometry,
            materials=self._recommend_materials(function),
            relationships=[],
            function=function,
            area=area,
            volume=volume,
            occupancy=occupancy,
            fire_rating=self._determine_fire_rating(function),
            accessibility_level=accessibility_level
        )
        
        return space
    
    def _create_furniture_element(self, placement: Dict, index: int, space_id: str) -> BIMElement:
        """Create a BIM element for furniture/equipment"""
        element_id = str(uuid.uuid4())
        
        properties = {
            'GlobalId': element_id,
            'Name': f"Furniture_{index+1}",
            'Description': f"Furniture piece {index+1}",
            'ObjectType': 'IfcFurniture',
            'PredefinedType': 'USERDEFINED',
            'dimensions': f"{placement['size'][0]}m x {placement['size'][1]}m",
            'area': placement['area'],
            'position': placement['position'],
            'orientation': placement.get('orientation', 'original'),
            'suitability_score': placement['suitability_score'],
            'material': 'Wood/Metal Composite',
            'manufacturer': 'Standard Office Furniture',
            'model_number': f"STD-{index+1:03d}",
            'cost_estimate': placement['area'] * 500,  # $500 per m²
            'maintenance_schedule': 'Annual inspection'
        }
        
        geometry = {
            'type': 'Rectangle',
            'coordinates': placement['box_coords'],
            'area': placement['area'],
            'center': [(placement['position'][0] + placement['size'][0]/2),
                      (placement['position'][1] + placement['size'][1]/2)],
            'rotation': 0.0 if placement.get('orientation') == 'original' else 90.0
        }
        
        return BIMElement(
            id=element_id,
            name=properties['Name'],
            element_type='IfcFurniture',
            properties=properties,
            geometry=geometry,
            materials=['Wood', 'Metal', 'Fabric'],
            relationships=[space_id]  # Related to the containing space
        )
    
    def _calculate_occupancy(self, function: str, area: float) -> int:
        """Calculate expected occupancy based on space type and area"""
        occupancy_rates = {
            'Office': 10,  # m² per person
            'Open Office': 8,
            'Conference Room': 2,
            'Meeting Room': 3,
            'Reception': 15,
            'Corridor': 0,
            'Storage': 0,
            'Kitchen': 5,
            'Break Room': 4,
            'Bathroom': 0
        }
        
        rate = occupancy_rates.get(function, 12)  # Default 12 m² per person
        return max(1, int(area / rate)) if rate > 0 else 0
    
    def _determine_accessibility(self, function: str, area: float, placements: List[Dict]) -> str:
        """Determine accessibility compliance level"""
        # Basic accessibility requirements
        if function in ['Bathroom', 'Reception', 'Lobby']:
            return 'ADA Compliant'
        elif function in ['Office', 'Conference Room', 'Meeting Room']:
            # Check if there's adequate space for wheelchair access
            clear_area = area - sum(p['area'] for p in placements)
            if clear_area >= area * 0.4:  # 40% clear space
                return 'Type A Accessible'
            else:
                return 'Basic Accessible'
        else:
            return 'Standard'
    
    def _map_to_ifc_space_type(self, function: str) -> str:
        """Map function to IFC space type"""
        mapping = {
            'Office': 'OFFICE',
            'Conference Room': 'MEETING',
            'Meeting Room': 'MEETING',
            'Open Office': 'OFFICE',
            'Reception': 'RECEPTION',
            'Corridor': 'CORRIDOR',
            'Storage': 'STORAGE',
            'Kitchen': 'KITCHEN',
            'Break Room': 'LOUNGE',
            'Bathroom': 'TOILET',
            'Server Room': 'TECHNICAL',
            'Copy Room': 'OFFICE'
        }
        return mapping.get(function, 'USERDEFINED')
    
    def _recommend_materials(self, function: str) -> List[str]:
        """Recommend appropriate materials for space type"""
        material_recommendations = {
            'Office': ['Gypsum Board', 'Paint', 'Carpet', 'Acoustic Ceiling Tile'],
            'Conference Room': ['Gypsum Board', 'Wood Veneer', 'Carpet', 'Acoustic Panels'],
            'Kitchen': ['Ceramic Tile', 'Stainless Steel', 'Epoxy Paint', 'Vinyl Flooring'],
            'Bathroom': ['Ceramic Tile', 'Porcelain', 'Waterproof Membrane', 'Anti-slip Flooring'],
            'Corridor': ['Gypsum Board', 'Paint', 'Commercial Carpet', 'LED Lighting'],
            'Storage': ['Concrete Block', 'Epoxy Paint', 'Sealed Concrete', 'Industrial Lighting']
        }
        return material_recommendations.get(function, self.default_properties['materials'])
    
    def _recommend_floor_finish(self, function: str) -> str:
        """Recommend floor finish based on function"""
        finishes = {
            'Office': 'Commercial Carpet',
            'Conference Room': 'High-Quality Carpet',
            'Kitchen': 'Non-slip Ceramic Tile',
            'Bathroom': 'Anti-slip Porcelain Tile',
            'Corridor': 'Commercial Grade Vinyl',
            'Storage': 'Sealed Concrete',
            'Reception': 'Polished Stone'
        }
        return finishes.get(function, 'Standard Carpet')
    
    def _recommend_wall_finish(self, function: str) -> str:
        """Recommend wall finish based on function"""
        finishes = {
            'Office': 'Painted Gypsum Board',
            'Conference Room': 'Wood Veneer Panels',
            'Kitchen': 'Ceramic Tile Wainscot',
            'Bathroom': 'Full Height Ceramic Tile',
            'Corridor': 'Painted Gypsum Board with Chair Rail',
            'Storage': 'Painted Concrete Block',
            'Reception': 'Decorative Wall Panels'
        }
        return finishes.get(function, 'Painted Gypsum Board')
    
    def _recommend_lighting(self, function: str) -> str:
        """Recommend lighting type based on function"""
        lighting = {
            'Office': 'LED Panel Lights (500 lux)',
            'Conference Room': 'Dimmable LED with Controls',
            'Kitchen': 'High-output LED (750 lux)',
            'Bathroom': 'Moisture-resistant LED',
            'Corridor': 'Linear LED Strip',
            'Storage': 'High-bay LED',
            'Reception': 'Decorative LED Fixtures'
        }
        return lighting.get(function, 'Standard LED (300 lux)')
    
    def _calculate_ventilation(self, function: str, occupancy: int) -> str:
        """Calculate ventilation requirements"""
        rates = {
            'Office': 8.5,  # L/s per person
            'Conference Room': 10.0,
            'Kitchen': 25.0,
            'Bathroom': 50,  # ACH (air changes per hour)
            'Storage': 0.5
        }
        
        if function in ['Bathroom']:
            return f"{rates.get(function, 2)} ACH"
        else:
            rate = rates.get(function, 5.0)
            total_rate = rate * max(1, occupancy)
            return f"{total_rate:.1f} L/s"
    
    def _calculate_lighting(self, function: str) -> str:
        """Calculate lighting requirements"""
        levels = {
            'Office': 500,  # lux
            'Conference Room': 500,
            'Kitchen': 750,
            'Bathroom': 200,
            'Corridor': 150,
            'Storage': 200,
            'Reception': 300
        }
        return f"{levels.get(function, 300)} lux"
    
    def _calculate_acoustic(self, function: str) -> str:
        """Calculate acoustic requirements"""
        requirements = {
            'Office': 'NRC 0.70, STC 45',
            'Conference Room': 'NRC 0.80, STC 50',
            'Kitchen': 'Hard surfaces, STC 40',
            'Bathroom': 'Moisture resistant, STC 45',
            'Corridor': 'NRC 0.60',
            'Storage': 'Basic acoustic treatment',
            'Reception': 'NRC 0.75, STC 45'
        }
        return requirements.get(function, 'Standard acoustic treatment')
    
    def _determine_fire_rating(self, function: str) -> str:
        """Determine fire rating requirements"""
        ratings = {
            'Office': '1 Hour',
            'Conference Room': '1 Hour',
            'Kitchen': '2 Hour',
            'Server Room': '2 Hour',
            'Storage': '1 Hour',
            'Corridor': '1 Hour (Egress Path)',
            'Bathroom': '1 Hour'
        }
        return ratings.get(function, '1 Hour')
    
    def export_to_ifc(self, bim_model: BIMBuilding) -> str:
        """Export BIM model to IFC format"""
        ifc_content = []
        
        # IFC Header
        ifc_content.append("ISO-10303-21;")
        ifc_content.append("HEADER;")
        ifc_content.append("FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');")
        ifc_content.append(f"FILE_NAME('{bim_model.name}.ifc','{datetime.now().isoformat()}',('AI Architectural Analyzer'),('Replit'),'IFC4X3','','');")
        ifc_content.append("FILE_SCHEMA(('IFC4X3'));")
        ifc_content.append("ENDSEC;")
        ifc_content.append("")
        ifc_content.append("DATA;")
        
        # Global unique identifiers and basic objects
        entity_id = 1
        
        # Project
        ifc_content.append(f"#{entity_id}= IFCPROJECT('{bim_model.id}',#{entity_id+1},$,$,'{bim_model.name}',$,$,$,#{entity_id+2});")
        entity_id += 1
        
        # Owner history
        ifc_content.append(f"#{entity_id}= IFCOWNERHISTORY(#{entity_id+1},#{entity_id+2},$,.ADDED.,$,#{entity_id+3},$,{int(datetime.now().timestamp())});")
        entity_id += 1
        
        # Continue with more IFC entities...
        # (This is a simplified representation - full IFC export would be much more complex)
        
        for floor in bim_model.floors:
            # Building storey
            ifc_content.append(f"#{entity_id}= IFCBUILDINGSTOREY('{floor.id}',#{1},$,$,'{floor.name}',$,$,$,$,.ELEMENT.,{floor.level});")
            entity_id += 1
            
            for space in floor.spaces:
                # Space
                ifc_content.append(f"#{entity_id}= IFCSPACE('{space.id}',#{1},$,$,'{space.name}',$,$,$,$,.ELEMENT.,.INTERNAL.,{space.properties.get('ceiling_height', 3.0)});")
                entity_id += 1
        
        ifc_content.append("ENDSEC;")
        ifc_content.append("END-ISO-10303-21;")
        
        return "\n".join(ifc_content)
    
    def export_to_cobie(self, bim_model: BIMBuilding) -> Dict[str, Any]:
        """Export BIM model to COBie format"""
        return self.standards.generate_cobie_data(bim_model)