import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

@dataclass
class FurnitureItem:
    """Individual furniture item with specifications"""
    id: str
    name: str
    category: str
    subcategory: str
    dimensions: Dict[str, float]  # width, height, depth in meters
    cost: float
    sustainability_score: float  # 0-1 scale
    brand: str
    material: str
    color_options: List[str]
    room_compatibility: List[str]
    required_clearance: Dict[str, float]  # clearances needed around item
    weight: float
    assembly_required: bool
    warranty_years: int

@dataclass
class FurnitureConfiguration:
    """Complete furniture configuration for a space"""
    space_id: str
    space_type: str
    items: List[FurnitureItem]
    layout_positions: List[Dict[str, float]]  # x, y, rotation for each item
    total_cost: float
    total_items: int
    sustainability_score: float
    space_utilization: float
    accessibility_compliant: bool

class FurnitureCatalogManager:
    """Professional furniture catalog with real product data"""

    def __init__(self, db_path: str = "furniture_catalog.db"):
        self.db_path = db_path
        self._initialize_database()
        self._populate_catalog()

    def _initialize_database(self):
        """Initialize furniture catalog database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create furniture items table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS furniture_items (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            width REAL NOT NULL,
            height REAL NOT NULL,
            depth REAL NOT NULL,
            cost REAL NOT NULL DEFAULT 0.0,
            sustainability_score REAL DEFAULT 0.5,
            brand TEXT,
            material TEXT,
            color_options TEXT,
            room_compatibility TEXT,
            clearance_front REAL DEFAULT 0.5,
            clearance_back REAL DEFAULT 0.3,
            clearance_sides REAL DEFAULT 0.3,
            weight REAL DEFAULT 10.0,
            assembly_required BOOLEAN DEFAULT 0,
            warranty_years INTEGER DEFAULT 1
        )
        ''')
        
        # Add missing columns if they don't exist
        cursor.execute("PRAGMA table_info(furniture_items)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'cost' not in columns:
            cursor.execute('ALTER TABLE furniture_items ADD COLUMN cost REAL DEFAULT 0.0')
        if 'sustainability_score' not in columns:
            cursor.execute('ALTER TABLE furniture_items ADD COLUMN sustainability_score REAL DEFAULT 0.5')

        # Create configurations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS furniture_configurations (
            id TEXT PRIMARY KEY,
            space_id TEXT NOT NULL,
            space_type TEXT NOT NULL,
            items_json TEXT NOT NULL,
            positions_json TEXT NOT NULL,
            total_cost REAL,
            total_items INTEGER,
            sustainability_score REAL,
            space_utilization REAL,
            accessibility_compliant BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()

    def _populate_catalog(self):
        """Populate catalog with real furniture data"""
        # Check if catalog is already populated
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM furniture_items")
        count = cursor.fetchone()[0]
        conn.close()

        if count == 0:
            self._add_office_furniture()
            self._add_residential_furniture()
            self._add_commercial_furniture()

    def _add_office_furniture(self):
        """Add professional office furniture items"""
        office_items = [
            # Desks
            FurnitureItem("DSK001", "Executive Desk", "Office", "Desks", 
                         {"width": 1.8, "height": 0.75, "depth": 0.9}, 1200.0, 0.7,
                         "Herman Miller", "Wood/Steel", ["Walnut", "Oak", "Black"],
                         ["Office", "Study"], {"front": 1.2, "back": 0.6, "sides": 0.3},
                         45.0, True, 10),

            FurnitureItem("DSK002", "Standing Desk", "Office", "Desks",
                         {"width": 1.4, "height": 1.1, "depth": 0.7}, 800.0, 0.8,
                         "Steelcase", "Metal/Wood", ["White", "Black", "Natural"],
                         ["Office"], {"front": 1.0, "back": 0.5, "sides": 0.3},
                         35.0, True, 7),

            # Chairs
            FurnitureItem("CHR001", "Ergonomic Office Chair", "Office", "Seating",
                         {"width": 0.65, "height": 1.2, "depth": 0.65}, 450.0, 0.6,
                         "Herman Miller", "Mesh/Plastic", ["Black", "Gray", "Blue"],
                         ["Office", "Conference Room"], {"front": 0.8, "back": 0.5, "sides": 0.3},
                         18.0, True, 12),

            FurnitureItem("CHR002", "Conference Chair", "Office", "Seating",
                         {"width": 0.6, "height": 0.9, "depth": 0.6}, 280.0, 0.7,
                         "Steelcase", "Leather/Steel", ["Black", "Brown", "Gray"],
                         ["Conference Room", "Meeting Room"], {"front": 0.6, "back": 0.4, "sides": 0.2},
                         15.0, False, 5),

            # Storage
            FurnitureItem("STR001", "Filing Cabinet", "Office", "Storage",
                         {"width": 0.4, "height": 1.3, "depth": 0.6}, 320.0, 0.5,
                         "Hon", "Steel", ["Gray", "Black", "White"],
                         ["Office"], {"front": 0.8, "back": 0.2, "sides": 0.2},
                         55.0, False, 15),

            # Tables
            FurnitureItem("TBL001", "Conference Table", "Office", "Tables",
                         {"width": 3.0, "height": 0.74, "depth": 1.2}, 1800.0, 0.8,
                         "Knoll", "Wood", ["Walnut", "Maple", "Cherry"],
                         ["Conference Room"], {"front": 1.0, "back": 1.0, "sides": 0.8},
                         80.0, True, 20)
        ]

        self._save_items_to_db(office_items)

    def _add_residential_furniture(self):
        """Add residential furniture items"""
        residential_items = [
            # Living Room
            FurnitureItem("SOF001", "3-Seater Sofa", "Residential", "Seating",
                         {"width": 2.1, "height": 0.85, "depth": 0.9}, 1200.0, 0.6,
                         "West Elm", "Fabric/Wood", ["Gray", "Blue", "Beige"],
                         ["Living Room"], {"front": 1.2, "back": 0.3, "sides": 0.3},
                         65.0, True, 3),

            FurnitureItem("TBL002", "Coffee Table", "Residential", "Tables",
                         {"width": 1.2, "height": 0.45, "depth": 0.6}, 400.0, 0.7,
                         "IKEA", "Wood", ["Oak", "Walnut", "White"],
                         ["Living Room"], {"front": 0.5, "back": 0.3, "sides": 0.3},
                         25.0, True, 2),

            # Bedroom
            FurnitureItem("BED001", "Queen Bed", "Residential", "Beds",
                         {"width": 1.6, "height": 0.9, "depth": 2.0}, 800.0, 0.5,
                         "IKEA", "Wood/Metal", ["Natural", "Black", "White"],
                         ["Bedroom"], {"front": 0.8, "back": 0.5, "sides": 0.6},
                         45.0, True, 5),

            FurnitureItem("WRD001", "Wardrobe", "Residential", "Storage",
                         {"width": 1.2, "height": 2.0, "depth": 0.6}, 600.0, 0.6,
                         "IKEA", "Wood", ["White", "Oak", "Black"],
                         ["Bedroom"], {"front": 0.8, "back": 0.1, "sides": 0.1},
                         70.0, True, 10)
        ]

        self._save_items_to_db(residential_items)

    def _add_commercial_furniture(self):
        """Add commercial furniture items"""
        commercial_items = [
            # Reception
            FurnitureItem("RCP001", "Reception Desk", "Commercial", "Reception",
                         {"width": 2.4, "height": 1.1, "depth": 0.8}, 2000.0, 0.7,
                         "Steelcase", "Wood/Steel", ["Walnut", "Maple"],
                         ["Lobby", "Reception"], {"front": 1.5, "back": 0.5, "sides": 0.3},
                         90.0, True, 15),

            # Waiting Area
            FurnitureItem("CHR003", "Lobby Chair", "Commercial", "Seating",
                         {"width": 0.75, "height": 0.8, "depth": 0.75}, 350.0, 0.8,
                         "Herman Miller", "Leather/Steel", ["Black", "Brown"],
                         ["Lobby", "Waiting Area"], {"front": 0.6, "back": 0.3, "sides": 0.3},
                         20.0, False, 8)
        ]

        self._save_items_to_db(commercial_items)

    def _save_items_to_db(self, items: List[FurnitureItem]):
        """Save furniture items to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for item in items:
            cursor.execute('''
            INSERT OR REPLACE INTO furniture_items VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ''', (
                item.id, item.name, item.category, item.subcategory,
                item.dimensions["width"], item.dimensions["height"], item.dimensions["depth"],
                item.cost, item.sustainability_score, item.brand, item.material,
                json.dumps(item.color_options), json.dumps(item.room_compatibility),
                item.required_clearance["front"], item.required_clearance["back"],
                item.required_clearance["sides"], item.weight, item.assembly_required,
                item.warranty_years
            ))

        conn.commit()
        conn.close()

    def recommend_furniture_for_space(self, space_type: str, space_area: float, 
                                    budget: float = 10000.0, 
                                    sustainability_preference: float = 0.5) -> FurnitureConfiguration:
        """Generate furniture recommendations for a space"""

        # Get compatible furniture items
        compatible_items = self._get_compatible_furniture(space_type, budget, sustainability_preference)

        # Select optimal furniture combination
        selected_items, positions = self._optimize_furniture_selection(
            compatible_items, space_area, budget, space_type
        )

        # Calculate metrics
        total_cost = sum(item.cost for item in selected_items)
        avg_sustainability = np.mean([item.sustainability_score for item in selected_items]) if selected_items else 0
        space_utilization = self._calculate_space_utilization(selected_items, positions, space_area)
        accessibility_compliant = self._check_accessibility_compliance(selected_items, positions, space_area)

        return FurnitureConfiguration(
            space_id=f"space_{hash(space_type + str(space_area))}",
            space_type=space_type,
            items=selected_items,
            layout_positions=positions,
            total_cost=total_cost,
            total_items=len(selected_items),
            sustainability_score=avg_sustainability,
            space_utilization=space_utilization,
            accessibility_compliant=accessibility_compliant
        )

    def _get_compatible_furniture(self, space_type: str, budget: float, 
                                sustainability_pref: float) -> List[FurnitureItem]:
        """Get furniture items compatible with space type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT * FROM furniture_items 
        WHERE cost <= ? AND sustainability_score >= ?
        ORDER BY sustainability_score DESC, cost ASC
        ''', (budget, sustainability_pref))

        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            try:
                item = FurnitureItem(
                    id=row[0], name=row[1], category=row[2], subcategory=row[3],
                    dimensions={"width": row[4], "height": row[5], "depth": row[6]},
                    cost=row[7], sustainability_score=row[8], brand=row[9], material=row[10],
                    color_options=json.loads(row[11]) if row[11] else [],
                    room_compatibility=json.loads(row[12]) if row[12] else [],
                    required_clearance={"front": row[13], "back": row[14], "sides": row[15]},
                    weight=row[16], assembly_required=bool(row[17]), warranty_years=row[18]
                )
                items.append(item)
            except (IndexError, json.JSONDecodeError) as e:
                # Skip malformed rows
                continue

        return items

    def _optimize_furniture_selection(self, available_items: List[FurnitureItem], 
                                    space_area: float, budget: float, 
                                    space_type: str) -> Tuple[List[FurnitureItem], List[Dict]]:
        """Select optimal furniture combination for space"""

        # Define essential furniture by space type
        essential_items = {
            'Office': ['Desks', 'Seating', 'Storage'],
            'Conference Room': ['Tables', 'Seating'],
            'Living Room': ['Seating', 'Tables'],
            'Bedroom': ['Beds', 'Storage'],
            'Kitchen': ['Tables', 'Storage'],
            'Lobby': ['Seating', 'Reception']
        }

        selected_items = []
        positions = []
        remaining_budget = budget
        used_area = 0

        # Prioritize essential items
        for essential_category in essential_items.get(space_type, []):
            for item in available_items:
                if (item.subcategory == essential_category and 
                    item.cost <= remaining_budget and
                    used_area + (item.dimensions["width"] * item.dimensions["depth"]) <= space_area * 0.6):

                    selected_items.append(item)
                    remaining_budget -= item.cost
                    used_area += item.dimensions["width"] * item.dimensions["depth"]

                    # Add position (simplified placement)
                    positions.append({
                        "x": len(positions) * 1.5,
                        "y": 1.0,
                        "rotation": 0,
                        "dimensions": {
                            "width": item.dimensions["width"],
                            "height": item.dimensions["height"],
                            "depth": item.dimensions["depth"]
                        }
                    })
                    break

        return selected_items, positions

    def _calculate_space_utilization(self, items: List[FurnitureItem], 
                                   positions: List[Dict], space_area: float) -> float:
        """Calculate how well the space is utilized"""
        if not items or space_area == 0:
            return 0.0

        furniture_area = sum(item.dimensions["width"] * item.dimensions["depth"] for item in items)
        return min(furniture_area / space_area, 1.0)

    def _check_accessibility_compliance(self, items: List[FurnitureItem], 
                                      positions: List[Dict], space_area: float) -> bool:
        """Check if furniture layout meets accessibility standards"""
        # Simplified accessibility check - ensure minimum clearance paths
        if not items:
            return True

        # Check if there's sufficient circulation space (minimum 30% of floor area)
        furniture_area = sum(item.dimensions["width"] * item.dimensions["depth"] for item in items)
        circulation_ratio = 1 - (furniture_area / space_area) if space_area > 0 else 0

        return circulation_ratio >= 0.3

    def get_furniture_by_category(self, category: str) -> List[FurnitureItem]:
        """Get all furniture items in a specific category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM furniture_items WHERE category = ?', (category,))
        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            try:
                item = FurnitureItem(
                    id=row[0], name=row[1], category=row[2], subcategory=row[3],
                    dimensions={"width": row[4], "height": row[5], "depth": row[6]},
                    cost=row[7], sustainability_score=row[8], brand=row[9], material=row[10],
                    color_options=json.loads(row[11]) if row[11] else [],
                    room_compatibility=json.loads(row[12]) if row[12] else [],
                    required_clearance={"front": row[13], "back": row[14], "sides": row[15]},
                    weight=row[16], assembly_required=bool(row[17]), warranty_years=row[18]
                )
                items.append(item)
            except (IndexError, json.JSONDecodeError) as e:
                # Skip malformed rows
                continue

        return items

    def save_configuration(self, config: FurnitureConfiguration):
        """Save furniture configuration to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO furniture_configurations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ''', (
            config.space_id, config.space_id, config.space_type,
            json.dumps([asdict(item) for item in config.items]),
            json.dumps(config.layout_positions),
            config.total_cost, config.total_items, config.sustainability_score,
            config.space_utilization, config.accessibility_compliant
        ))

        conn.commit()
        conn.close()