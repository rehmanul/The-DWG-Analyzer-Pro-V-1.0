import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon, box
import requests
import sqlite3

@dataclass
class FurnitureItem:
    """Represents a furniture catalog item"""
    item_id: str
    name: str
    category: str
    subcategory: str
    manufacturer: str
    model_number: str
    dimensions: Tuple[float, float, float]  # length, width, height
    weight: float
    material: str
    color_options: List[str]
    price: float
    currency: str
    availability: str
    lead_time_days: int
    description: str
    specifications: Dict[str, Any]
    sustainability_rating: str
    certifications: List[str]
    image_urls: List[str]
    cad_file_url: Optional[str]
    last_updated: datetime

@dataclass
class PricingTier:
    """Pricing structure for bulk orders"""
    min_quantity: int
    max_quantity: Optional[int]
    unit_price: float
    discount_percentage: float

@dataclass
class FurnitureConfiguration:
    """Complete furniture configuration for a space"""
    config_id: str
    space_type: str
    total_items: int
    total_cost: float
    items: List[Dict[str, Any]]  # item details with quantities and positions
    space_utilization: float
    ergonomic_score: float
    sustainability_score: float
    estimated_delivery: datetime

class FurnitureCatalogManager:
    """
    Comprehensive furniture catalog integration with pricing and availability
    """
    
    def __init__(self, catalog_db_path: str = "furniture_catalog.db"):
        self.db_path = catalog_db_path
        self.init_database()
        self.load_standard_catalog()
        
        # Furniture categories and their typical dimensions
        self.furniture_categories = {
            'Desk': {
                'standard_sizes': [(1.2, 0.6, 0.75), (1.4, 0.7, 0.75), (1.6, 0.8, 0.75)],
                'price_range': (200, 2000),
                'space_requirements': {'clearance': 0.6, 'approach': 0.9}
            },
            'Chair': {
                'standard_sizes': [(0.6, 0.6, 0.9), (0.65, 0.65, 1.1)],
                'price_range': (100, 1500),
                'space_requirements': {'clearance': 0.3, 'approach': 0.6}
            },
            'Conference Table': {
                'standard_sizes': [(2.4, 1.2, 0.75), (3.0, 1.2, 0.75), (3.6, 1.2, 0.75)],
                'price_range': (800, 5000),
                'space_requirements': {'clearance': 0.9, 'approach': 1.2}
            },
            'Storage Cabinet': {
                'standard_sizes': [(0.8, 0.4, 1.8), (1.0, 0.4, 2.0), (1.2, 0.5, 2.0)],
                'price_range': (300, 1200),
                'space_requirements': {'clearance': 0.2, 'approach': 0.8}
            },
            'Workstation': {
                'standard_sizes': [(1.5, 1.5, 0.75), (1.8, 1.2, 0.75), (2.0, 1.5, 0.75)],
                'price_range': (500, 3000),
                'space_requirements': {'clearance': 0.8, 'approach': 1.0}
            },
            'Meeting Chair': {
                'standard_sizes': [(0.55, 0.55, 0.8), (0.6, 0.6, 0.85)],
                'price_range': (80, 800),
                'space_requirements': {'clearance': 0.2, 'approach': 0.5}
            },
            'Sofa': {
                'standard_sizes': [(1.8, 0.9, 0.8), (2.1, 0.9, 0.8), (2.4, 0.9, 0.8)],
                'price_range': (600, 3000),
                'space_requirements': {'clearance': 0.5, 'approach': 1.0}
            },
            'Coffee Table': {
                'standard_sizes': [(1.0, 0.5, 0.4), (1.2, 0.6, 0.4), (1.4, 0.7, 0.4)],
                'price_range': (150, 800),
                'space_requirements': {'clearance': 0.4, 'approach': 0.6}
            }
        }
        
        # Sustainability ratings
        self.sustainability_criteria = {
            'A+': {'recycled_content': 75, 'renewable_materials': 80, 'low_emissions': True},
            'A': {'recycled_content': 50, 'renewable_materials': 60, 'low_emissions': True},
            'B': {'recycled_content': 25, 'renewable_materials': 40, 'low_emissions': False},
            'C': {'recycled_content': 10, 'renewable_materials': 20, 'low_emissions': False}
        }
    
    def init_database(self):
        """Initialize the furniture catalog database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create furniture items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS furniture_items (
                item_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                manufacturer TEXT,
                model_number TEXT,
                length REAL,
                width REAL,
                height REAL,
                weight REAL,
                material TEXT,
                color_options TEXT,
                base_price REAL,
                currency TEXT,
                availability TEXT,
                lead_time_days INTEGER,
                description TEXT,
                specifications TEXT,
                sustainability_rating TEXT,
                certifications TEXT,
                image_urls TEXT,
                cad_file_url TEXT,
                last_updated TEXT
            )
        ''')
        
        # Create pricing tiers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pricing_tiers (
                tier_id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                min_quantity INTEGER,
                max_quantity INTEGER,
                unit_price REAL,
                discount_percentage REAL,
                FOREIGN KEY (item_id) REFERENCES furniture_items (item_id)
            )
        ''')
        
        # Create configurations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS furniture_configurations (
                config_id TEXT PRIMARY KEY,
                space_type TEXT,
                total_items INTEGER,
                total_cost REAL,
                items_json TEXT,
                space_utilization REAL,
                ergonomic_score REAL,
                sustainability_score REAL,
                estimated_delivery TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_standard_catalog(self):
        """Load standard furniture catalog with realistic items"""
        standard_items = [
            # Desks
            {
                'item_id': 'DSK001',
                'name': 'Executive Desk - Premium',
                'category': 'Desk',
                'subcategory': 'Executive',
                'manufacturer': 'Herman Miller',
                'model_number': 'HM-EXE-160',
                'dimensions': (1.6, 0.8, 0.75),
                'weight': 45.0,
                'material': 'Oak Veneer with Steel Frame',
                'color_options': ['Natural Oak', 'Walnut', 'Espresso'],
                'price': 1200.0,
                'availability': 'In Stock',
                'lead_time_days': 14,
                'description': 'Premium executive desk with cable management and locking drawers',
                'sustainability_rating': 'A',
                'certifications': ['GREENGUARD Gold', 'FSC Certified']
            },
            {
                'item_id': 'DSK002',
                'name': 'Compact Workstation',
                'category': 'Desk',
                'subcategory': 'Workstation',
                'manufacturer': 'Steelcase',
                'model_number': 'SC-CWS-120',
                'dimensions': (1.2, 0.6, 0.75),
                'weight': 28.0,
                'material': 'Laminate with Metal Legs',
                'color_options': ['White', 'Gray', 'Black'],
                'price': 450.0,
                'availability': 'In Stock',
                'lead_time_days': 7,
                'description': 'Space-efficient workstation perfect for modern offices',
                'sustainability_rating': 'B',
                'certifications': ['GREENGUARD']
            },
            # Chairs
            {
                'item_id': 'CHR001',
                'name': 'Ergonomic Task Chair',
                'category': 'Chair',
                'subcategory': 'Task',
                'manufacturer': 'Herman Miller',
                'model_number': 'HM-AERON-B',
                'dimensions': (0.65, 0.65, 1.1),
                'weight': 18.0,
                'material': 'Polymer Mesh with Aluminum Frame',
                'color_options': ['Graphite', 'Carbon', 'Mineral'],
                'price': 895.0,
                'availability': 'In Stock',
                'lead_time_days': 10,
                'description': 'Award-winning ergonomic chair with advanced PostureFit support',
                'sustainability_rating': 'A+',
                'certifications': ['GREENGUARD Gold', 'Cradle to Cradle Bronze']
            },
            {
                'item_id': 'CHR002',
                'name': 'Conference Chair - Leather',
                'category': 'Chair',
                'subcategory': 'Conference',
                'manufacturer': 'Knoll',
                'model_number': 'KN-CONF-LTH',
                'dimensions': (0.6, 0.6, 0.9),
                'weight': 12.0,
                'material': 'Genuine Leather with Chrome Base',
                'color_options': ['Black', 'Brown', 'Tan'],
                'price': 650.0,
                'availability': 'Limited Stock',
                'lead_time_days': 21,
                'description': 'Premium leather conference chair with swivel base',
                'sustainability_rating': 'B',
                'certifications': ['GREENGUARD']
            },
            # Conference Tables
            {
                'item_id': 'TBL001',
                'name': 'Oval Conference Table',
                'category': 'Conference Table',
                'subcategory': 'Oval',
                'manufacturer': 'Steelcase',
                'model_number': 'SC-OVAL-300',
                'dimensions': (3.0, 1.2, 0.75),
                'weight': 85.0,
                'material': 'Solid Wood Top with Steel Base',
                'color_options': ['Cherry', 'Maple', 'Walnut'],
                'price': 2400.0,
                'availability': 'Made to Order',
                'lead_time_days': 42,
                'description': '12-person oval conference table with integrated power/data',
                'sustainability_rating': 'A',
                'certifications': ['FSC Certified', 'GREENGUARD Gold']
            },
            # Storage
            {
                'item_id': 'STG001',
                'name': 'Modular Storage System',
                'category': 'Storage Cabinet',
                'subcategory': 'Modular',
                'manufacturer': 'Teknion',
                'model_number': 'TK-MOD-180',
                'dimensions': (1.0, 0.4, 1.8),
                'weight': 45.0,
                'material': 'Metal Frame with Wood Shelves',
                'color_options': ['White/Oak', 'Black/Walnut', 'Gray/Maple'],
                'price': 780.0,
                'availability': 'In Stock',
                'lead_time_days': 14,
                'description': 'Versatile modular storage with adjustable shelves',
                'sustainability_rating': 'A',
                'certifications': ['GREENGUARD', 'FSC Certified']
            },
            # Workstations
            {
                'item_id': 'WKS001',
                'name': 'Open Plan Workstation',
                'category': 'Workstation',
                'subcategory': 'Open Plan',
                'manufacturer': 'Herman Miller',
                'model_number': 'HM-OPN-150',
                'dimensions': (1.5, 1.5, 0.75),
                'weight': 65.0,
                'material': 'Laminate Surfaces with Fabric Panels',
                'color_options': ['Light Gray', 'Navy', 'Sage Green'],
                'price': 1100.0,
                'availability': 'In Stock',
                'lead_time_days': 21,
                'description': 'Complete workstation with privacy panels and storage',
                'sustainability_rating': 'A',
                'certifications': ['GREENGUARD Gold', 'SCS Certified']
            }
        ]
        
        # Insert standard items into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in standard_items:
            cursor.execute('''
                INSERT OR REPLACE INTO furniture_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['item_id'], item['name'], item['category'], item.get('subcategory', ''),
                item['manufacturer'], item['model_number'],
                item['dimensions'][0], item['dimensions'][1], item['dimensions'][2],
                item['weight'], item['material'], json.dumps(item['color_options']),
                item['price'], 'USD', item['availability'], item['lead_time_days'],
                item['description'], '{}', item['sustainability_rating'],
                json.dumps(item['certifications']), '[]', '', datetime.now().isoformat()
            ))
            
            # Add pricing tiers
            self._add_pricing_tiers(cursor, item['item_id'], item['price'])
        
        conn.commit()
        conn.close()
    
    def _add_pricing_tiers(self, cursor, item_id: str, base_price: float):
        """Add volume pricing tiers for an item"""
        tiers = [
            (1, 9, base_price, 0.0),
            (10, 24, base_price * 0.95, 5.0),
            (25, 49, base_price * 0.90, 10.0),
            (50, 99, base_price * 0.85, 15.0),
            (100, None, base_price * 0.80, 20.0)
        ]
        
        for min_qty, max_qty, unit_price, discount in tiers:
            cursor.execute('''
                INSERT INTO pricing_tiers (item_id, min_quantity, max_quantity, unit_price, discount_percentage)
                VALUES (?, ?, ?, ?, ?)
            ''', (item_id, min_qty, max_qty, unit_price, discount))
    
    def search_furniture(self, category: Optional[str] = None, 
                        max_price: Optional[float] = None,
                        dimensions_max: Optional[Tuple[float, float, float]] = None,
                        sustainability_min: Optional[str] = None) -> List[FurnitureItem]:
        """Search furniture catalog with filters"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM furniture_items WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if max_price:
            query += " AND base_price <= ?"
            params.append(max_price)
        
        if dimensions_max:
            query += " AND length <= ? AND width <= ? AND height <= ?"
            params.extend(dimensions_max)
        
        if sustainability_min:
            # Simple sustainability filtering (A+ > A > B > C)
            sustainability_order = {'A+': 4, 'A': 3, 'B': 2, 'C': 1}
            min_rating = sustainability_order.get(sustainability_min, 1)
            
            sustainability_filter = " AND ("
            valid_ratings = [rating for rating, value in sustainability_order.items() if value >= min_rating]
            sustainability_filter += " OR ".join(["sustainability_rating = ?" for _ in valid_ratings])
            sustainability_filter += ")"
            
            query += sustainability_filter
            params.extend(valid_ratings)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        items = []
        for row in results:
            item = FurnitureItem(
                item_id=row[0],
                name=row[1],
                category=row[2],
                subcategory=row[3],
                manufacturer=row[4],
                model_number=row[5],
                dimensions=(row[6], row[7], row[8]),
                weight=row[9],
                material=row[10],
                color_options=json.loads(row[11]),
                price=row[12],
                currency=row[13],
                availability=row[14],
                lead_time_days=row[15],
                description=row[16],
                specifications=json.loads(row[17]),
                sustainability_rating=row[18],
                certifications=json.loads(row[19]),
                image_urls=json.loads(row[20]),
                cad_file_url=row[21],
                last_updated=datetime.fromisoformat(row[22])
            )
            items.append(item)
        
        conn.close()
        return items
    
    def get_pricing_for_quantity(self, item_id: str, quantity: int) -> Dict[str, Any]:
        """Get pricing information for specific quantity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT unit_price, discount_percentage FROM pricing_tiers 
            WHERE item_id = ? AND min_quantity <= ? AND (max_quantity IS NULL OR max_quantity >= ?)
            ORDER BY min_quantity DESC LIMIT 1
        ''', (item_id, quantity, quantity))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            unit_price, discount = result
            total_cost = unit_price * quantity
            total_savings = (unit_price * (discount / 100)) * quantity
            
            return {
                'unit_price': unit_price,
                'quantity': quantity,
                'discount_percentage': discount,
                'total_cost': total_cost,
                'total_savings': total_savings,
                'cost_per_item_after_discount': unit_price * (1 - discount / 100)
            }
        
        return {}
    
    def recommend_furniture_for_space(self, space_type: str, space_area: float, 
                                    budget: Optional[float] = None,
                                    sustainability_preference: Optional[str] = None) -> FurnitureConfiguration:
        """Recommend optimal furniture configuration for a space"""
        
        # Define space requirements based on type
        space_requirements = {
            'Office': {
                'essential': [('Desk', 1), ('Chair', 1)],
                'optional': [('Storage Cabinet', 1), ('Meeting Chair', 2)],
                'area_per_person': 10.0
            },
            'Conference Room': {
                'essential': [('Conference Table', 1)],
                'optional': [('Meeting Chair', 8), ('Storage Cabinet', 1)],
                'area_per_person': 2.5
            },
            'Open Office': {
                'essential': [('Workstation', 1), ('Chair', 1)],
                'optional': [('Storage Cabinet', 1), ('Meeting Chair', 1)],
                'area_per_person': 8.0
            },
            'Break Room': {
                'essential': [('Sofa', 1), ('Coffee Table', 1)],
                'optional': [('Chair', 4), ('Storage Cabinet', 1)],
                'area_per_person': 4.0
            }
        }
        
        space_req = space_requirements.get(space_type, space_requirements['Office'])
        estimated_occupancy = max(1, int(space_area / space_req['area_per_person']))
        
        # Get furniture recommendations
        recommended_items = []
        total_cost = 0.0
        
        # Essential furniture
        for category, base_quantity in space_req['essential']:
            quantity = base_quantity * estimated_occupancy if category != 'Conference Table' else 1
            items = self.search_furniture(category=category, max_price=budget)
            
            if items:
                # Select best item based on criteria
                best_item = self._select_best_item(items, budget, sustainability_preference)
                pricing = self.get_pricing_for_quantity(best_item.item_id, quantity)
                
                recommended_items.append({
                    'item': best_item,
                    'quantity': quantity,
                    'pricing': pricing,
                    'essential': True
                })
                
                total_cost += pricing.get('total_cost', 0)
        
        # Optional furniture (if budget allows)
        if budget is None or total_cost < budget * 0.8:  # Use 80% of budget for essentials
            remaining_budget = (budget - total_cost) if budget else None
            
            for category, base_quantity in space_req['optional']:
                if remaining_budget and remaining_budget < 200:  # Minimum for optional items
                    break
                
                quantity = base_quantity
                items = self.search_furniture(category=category, max_price=remaining_budget)
                
                if items:
                    best_item = self._select_best_item(items, remaining_budget, sustainability_preference)
                    pricing = self.get_pricing_for_quantity(best_item.item_id, quantity)
                    
                    item_cost = pricing.get('total_cost', 0)
                    if not remaining_budget or item_cost <= remaining_budget:
                        recommended_items.append({
                            'item': best_item,
                            'quantity': quantity,
                            'pricing': pricing,
                            'essential': False
                        })
                        
                        total_cost += item_cost
                        if remaining_budget:
                            remaining_budget -= item_cost
        
        # Calculate metrics
        total_items = sum(item['quantity'] for item in recommended_items)
        space_utilization = self._calculate_space_utilization(recommended_items, space_area)
        ergonomic_score = self._calculate_ergonomic_score(recommended_items)
        sustainability_score = self._calculate_sustainability_score(recommended_items)
        
        # Estimate delivery date
        max_lead_time = max(item['item'].lead_time_days for item in recommended_items)
        estimated_delivery = datetime.now()
        estimated_delivery = estimated_delivery.replace(day=estimated_delivery.day + max_lead_time)
        
        config = FurnitureConfiguration(
            config_id=f"CONFIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            space_type=space_type,
            total_items=total_items,
            total_cost=total_cost,
            items=recommended_items,
            space_utilization=space_utilization,
            ergonomic_score=ergonomic_score,
            sustainability_score=sustainability_score,
            estimated_delivery=estimated_delivery
        )
        
        # Save configuration
        self._save_configuration(config)
        
        return config
    
    def _select_best_item(self, items: List[FurnitureItem], budget: Optional[float],
                         sustainability_preference: Optional[str]) -> FurnitureItem:
        """Select the best item based on criteria"""
        
        # Filter by budget
        if budget:
            items = [item for item in items if item.price <= budget]
        
        if not items:
            return items[0] if items else None
        
        # Score items
        scored_items = []
        for item in items:
            score = 0.0
            
            # Price score (lower price = higher score)
            max_price = max(i.price for i in items)
            min_price = min(i.price for i in items)
            if max_price > min_price:
                price_score = 1.0 - ((item.price - min_price) / (max_price - min_price))
            else:
                price_score = 1.0
            score += price_score * 0.3
            
            # Sustainability score
            sustainability_scores = {'A+': 1.0, 'A': 0.8, 'B': 0.6, 'C': 0.4}
            sustainability_score = sustainability_scores.get(item.sustainability_rating, 0.2)
            
            if sustainability_preference:
                weight = 0.4 if sustainability_preference in ['A+', 'A'] else 0.2
            else:
                weight = 0.2
            
            score += sustainability_score * weight
            
            # Availability score
            availability_scores = {'In Stock': 1.0, 'Limited Stock': 0.7, 'Made to Order': 0.5}
            availability_score = availability_scores.get(item.availability, 0.3)
            score += availability_score * 0.2
            
            # Lead time score (shorter = better)
            max_lead_time = max(i.lead_time_days for i in items)
            min_lead_time = min(i.lead_time_days for i in items)
            if max_lead_time > min_lead_time:
                lead_time_score = 1.0 - ((item.lead_time_days - min_lead_time) / (max_lead_time - min_lead_time))
            else:
                lead_time_score = 1.0
            score += lead_time_score * 0.1
            
            # Brand/quality score (simplified)
            quality_scores = {'Herman Miller': 1.0, 'Steelcase': 0.9, 'Knoll': 0.9, 'Teknion': 0.8}
            quality_score = quality_scores.get(item.manufacturer, 0.7)
            score += quality_score * 0.2
            
            scored_items.append((item, score))
        
        # Return highest scoring item
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[0][0]
    
    def _calculate_space_utilization(self, items: List[Dict], space_area: float) -> float:
        """Calculate how efficiently the space is utilized"""
        total_furniture_area = 0.0
        
        for item_data in items:
            item = item_data['item']
            quantity = item_data['quantity']
            
            # Calculate furniture footprint (length × width)
            furniture_area = item.dimensions[0] * item.dimensions[1]
            total_furniture_area += furniture_area * quantity
        
        # Include circulation space (typically 40-60% of total area)
        utilized_area = total_furniture_area * 1.5  # Assumes 50% circulation
        
        return min(1.0, utilized_area / space_area) if space_area > 0 else 0.0
    
    def _calculate_ergonomic_score(self, items: List[Dict]) -> float:
        """Calculate ergonomic quality score"""
        total_score = 0.0
        total_weight = 0.0
        
        # Ergonomic weights by category
        ergonomic_weights = {
            'Chair': 0.4,
            'Desk': 0.3,
            'Workstation': 0.35,
            'Conference Table': 0.2,
            'Storage Cabinet': 0.1
        }
        
        for item_data in items:
            item = item_data['item']
            quantity = item_data['quantity']
            
            # Basic ergonomic scoring based on manufacturer and category
            manufacturer_scores = {'Herman Miller': 0.9, 'Steelcase': 0.85, 'Knoll': 0.8, 'Teknion': 0.75}
            base_score = manufacturer_scores.get(item.manufacturer, 0.7)
            
            # Adjust for category-specific ergonomic importance
            weight = ergonomic_weights.get(item.category, 0.1) * quantity
            total_score += base_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.7
    
    def _calculate_sustainability_score(self, items: List[Dict]) -> float:
        """Calculate overall sustainability score"""
        total_score = 0.0
        total_weight = 0.0
        
        sustainability_scores = {'A+': 1.0, 'A': 0.8, 'B': 0.6, 'C': 0.4}
        
        for item_data in items:
            item = item_data['item']
            quantity = item_data['quantity']
            
            score = sustainability_scores.get(item.sustainability_rating, 0.2)
            
            # Weight by item cost (more expensive items have more impact)
            weight = item.price * quantity
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _save_configuration(self, config: FurnitureConfiguration):
        """Save furniture configuration to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert items to JSON for storage
        items_json = json.dumps([{
            'item_id': item['item'].item_id,
            'quantity': item['quantity'],
            'pricing': item['pricing'],
            'essential': item['essential']
        } for item in config.items])
        
        cursor.execute('''
            INSERT INTO furniture_configurations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            config.config_id, config.space_type, config.total_items, config.total_cost,
            items_json, config.space_utilization, config.ergonomic_score,
            config.sustainability_score, config.estimated_delivery.isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_configuration(self, config_id: str) -> Optional[FurnitureConfiguration]:
        """Retrieve saved furniture configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM furniture_configurations WHERE config_id = ?', (config_id,))
        result = cursor.fetchone()
        
        if result:
            # Reconstruct the configuration
            items_data = json.loads(result[4])
            items = []
            
            for item_data in items_data:
                # Get full item details
                cursor.execute('SELECT * FROM furniture_items WHERE item_id = ?', (item_data['item_id'],))
                item_row = cursor.fetchone()
                
                if item_row:
                    item = FurnitureItem(
                        item_id=item_row[0],
                        name=item_row[1],
                        category=item_row[2],
                        subcategory=item_row[3],
                        manufacturer=item_row[4],
                        model_number=item_row[5],
                        dimensions=(item_row[6], item_row[7], item_row[8]),
                        weight=item_row[9],
                        material=item_row[10],
                        color_options=json.loads(item_row[11]),
                        price=item_row[12],
                        currency=item_row[13],
                        availability=item_row[14],
                        lead_time_days=item_row[15],
                        description=item_row[16],
                        specifications=json.loads(item_row[17]),
                        sustainability_rating=item_row[18],
                        certifications=json.loads(item_row[19]),
                        image_urls=json.loads(item_row[20]),
                        cad_file_url=item_row[21],
                        last_updated=datetime.fromisoformat(item_row[22])
                    )
                    
                    items.append({
                        'item': item,
                        'quantity': item_data['quantity'],
                        'pricing': item_data['pricing'],
                        'essential': item_data['essential']
                    })
            
            config = FurnitureConfiguration(
                config_id=result[0],
                space_type=result[1],
                total_items=result[2],
                total_cost=result[3],
                items=items,
                space_utilization=result[5],
                ergonomic_score=result[6],
                sustainability_score=result[7],
                estimated_delivery=datetime.fromisoformat(result[8])
            )
            
            conn.close()
            return config
        
        conn.close()
        return None
    
    def generate_procurement_report(self, config_id: str) -> Dict[str, Any]:
        """Generate comprehensive procurement report"""
        config = self.get_configuration(config_id)
        if not config:
            return {}
        
        # Group items by manufacturer for procurement efficiency
        by_manufacturer = {}
        for item_data in config.items:
            item = item_data['item']
            manufacturer = item.manufacturer
            
            if manufacturer not in by_manufacturer:
                by_manufacturer[manufacturer] = []
            
            by_manufacturer[manufacturer].append({
                'item': item,
                'quantity': item_data['quantity'],
                'total_cost': item_data['pricing']['total_cost'],
                'lead_time': item.lead_time_days
            })
        
        # Calculate delivery schedule
        delivery_schedule = []
        for manufacturer, items in by_manufacturer.items():
            max_lead_time = max(item['lead_time'] for item in items)
            total_cost = sum(item['total_cost'] for item in items)
            
            delivery_schedule.append({
                'manufacturer': manufacturer,
                'items_count': len(items),
                'total_cost': total_cost,
                'estimated_delivery_days': max_lead_time,
                'items': items
            })
        
        # Sort by delivery time
        delivery_schedule.sort(key=lambda x: x['estimated_delivery_days'])
        
        report = {
            'configuration_id': config_id,
            'space_type': config.space_type,
            'summary': {
                'total_items': config.total_items,
                'total_cost': config.total_cost,
                'unique_manufacturers': len(by_manufacturer),
                'estimated_delivery_days': max(d['estimated_delivery_days'] for d in delivery_schedule),
                'space_utilization': config.space_utilization,
                'ergonomic_score': config.ergonomic_score,
                'sustainability_score': config.sustainability_score
            },
            'delivery_schedule': delivery_schedule,
            'cost_breakdown': {
                'essential_items': sum(
                    item_data['pricing']['total_cost'] 
                    for item_data in config.items 
                    if item_data['essential']
                ),
                'optional_items': sum(
                    item_data['pricing']['total_cost'] 
                    for item_data in config.items 
                    if not item_data['essential']
                ),
                'total_savings': sum(
                    item_data['pricing'].get('total_savings', 0) 
                    for item_data in config.items
                )
            },
            'recommendations': self._generate_procurement_recommendations(config, by_manufacturer)
        }
        
        return report
    
    def _generate_procurement_recommendations(self, config: FurnitureConfiguration, 
                                           by_manufacturer: Dict) -> List[str]:
        """Generate procurement recommendations"""
        recommendations = []
        
        # Check for potential volume discounts
        for manufacturer, items in by_manufacturer.items():
            total_cost = sum(item['total_cost'] for item in items)
            if total_cost > 10000:  # $10k threshold
                recommendations.append(
                    f"Consider negotiating additional volume discount with {manufacturer} "
                    f"for ${total_cost:,.0f} order"
                )
        
        # Check delivery schedule optimization
        delivery_times = [d['estimated_delivery_days'] for d in by_manufacturer.values()]
        if max(delivery_times) - min(delivery_times) > 14:  # 2+ week difference
            recommendations.append(
                "Consider coordinating delivery schedules to reduce installation disruption"
            )
        
        # Sustainability recommendations
        if config.sustainability_score < 0.7:
            recommendations.append(
                "Consider upgrading to more sustainable options to improve environmental impact"
            )
        
        # Space utilization recommendations
        if config.space_utilization < 0.6:
            recommendations.append(
                "Space appears underutilized - consider adding functional furniture or storage"
            )
        elif config.space_utilization > 0.9:
            recommendations.append(
                "Space may be overcrowded - consider reducing furniture quantity or selecting smaller items"
            )
        
        return recommendations