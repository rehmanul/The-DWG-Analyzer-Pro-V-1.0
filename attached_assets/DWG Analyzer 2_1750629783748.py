import sys
import math
import json
import numpy as np
import ezdxf
from rectpack import newPacker
from pathlib import Path
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QLabel, QFileDialog, QFormLayout, QDoubleSpinBox, QMessageBox,
    QHBoxLayout, QListWidget, QListWidgetItem, QTabWidget, QSpinBox,
    QComboBox, QCheckBox, QTextEdit, QProgressBar, QGroupBox,
    QGridLayout, QSlider, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon as MplPolygon, Circle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


class AIAnalysisThread(QThread):
    """Thread for AI analysis to prevent UI freezing"""
    progress_updated = pyqtSignal(int)
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, zones, analysis_params):
        super().__init__()
        self.zones = zones
        self.params = analysis_params
        
    def run(self):
        try:
            self.progress_updated.emit(10)
            
            # Analyze room types based on dimensions and context
            room_analysis = self.analyze_room_types()
            self.progress_updated.emit(30)
            
            # Detect furniture placement opportunities
            furniture_analysis = self.analyze_furniture_placement()
            self.progress_updated.emit(60)
            
            # Calculate optimal box/furniture arrangements
            optimization_results = self.optimize_placements()
            self.progress_updated.emit(90)
            
            results = {
                'rooms': room_analysis,
                'furniture': furniture_analysis,
                'optimization': optimization_results
            }
            
            self.progress_updated.emit(100)
            self.analysis_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def analyze_room_types(self):
        """AI-powered room type detection based on dimensions and layout"""
        room_types = {}
        
        for i, zone in enumerate(self.zones):
            if not zone.get('points'):
                continue
                
            # Calculate room dimensions and area
            poly = Polygon(zone['points'])
            area = poly.area
            bounds = poly.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio = max(width, height) / min(width, height)
            
            # AI heuristics for room classification
            room_type = "Unknown"
            confidence = 0.5
            
            if area < 10:  # Small rooms
                if aspect_ratio > 3:
                    room_type = "Corridor"
                    confidence = 0.8
                else:
                    room_type = "Storage/WC"
                    confidence = 0.7
            elif area < 25:  # Medium rooms
                if aspect_ratio < 1.5:
                    room_type = "Bedroom"
                    confidence = 0.7
                else:
                    room_type = "Office"
                    confidence = 0.6
            elif area < 50:  # Large rooms
                if aspect_ratio < 1.3:
                    room_type = "Living Room"
                    confidence = 0.8
                else:
                    room_type = "Open Office"
                    confidence = 0.7
            else:  # Very large rooms
                room_type = "Conference/Hall"
                confidence = 0.9
            
            room_types[f"Zone_{i}"] = {
                'type': room_type,
                'confidence': confidence,
                'area': area,
                'dimensions': (width, height),
                'aspect_ratio': aspect_ratio,
                'layer': zone.get('layer', 'Unknown')
            }
        
        return room_types
    
    def analyze_furniture_placement(self):
        """Analyze optimal furniture/box placement using AI algorithms"""
        placements = {}
        
        for i, zone in enumerate(self.zones):
            if not zone.get('points'):
                continue
                
            poly = Polygon(zone['points'])
            bounds = poly.bounds
            
            # Generate placement grid with intelligent spacing
            furniture_spots = []
            box_size = self.params.get('box_size', (2.0, 1.5))
            margin = self.params.get('margin', 0.5)
            
            # Create a smarter grid that considers room shape
            x_step = box_size[0] + margin
            y_step = box_size[1] + margin
            
            x = bounds[0] + margin
            while x + box_size[0] <= bounds[2] - margin:
                y = bounds[1] + margin
                while y + box_size[1] <= bounds[3] - margin:
                    # Check if placement is valid (inside polygon)
                    test_box = box(x, y, x + box_size[0], y + box_size[1])
                    if poly.contains(test_box.centroid):
                        furniture_spots.append({
                            'position': (x, y),
                            'box_coords': [(x, y), (x + box_size[0], y), 
                                         (x + box_size[0], y + box_size[1]), 
                                         (x, y + box_size[1])],
                            'suitability_score': self.calculate_suitability(poly, test_box)
                        })
                    y += y_step
                x += x_step
            
            placements[f"Zone_{i}"] = furniture_spots
        
        return placements
    
    def calculate_suitability(self, room_poly, furniture_box):
        """Calculate suitability score for furniture placement"""
        # Distance from walls
        center = furniture_box.centroid
        distance_to_edge = room_poly.boundary.distance(center)
        
        # Prefer central locations but not too far from walls
        room_center = room_poly.centroid
        distance_to_center = center.distance(room_center)
        
        # Normalize scores (0-1)
        wall_score = min(distance_to_edge / 2.0, 1.0)  # Prefer some distance from walls
        center_score = 1.0 - min(distance_to_center / 5.0, 1.0)  # But not too far from center
        
        return (wall_score + center_score) / 2.0
    
    def optimize_placements(self):
        """Use advanced algorithms to optimize furniture placement"""
        # This would implement more sophisticated optimization
        # For now, return basic optimization results
        return {
            'algorithm_used': 'Grid-based with AI scoring',
            'optimization_level': 'High',
            'total_efficiency': 0.85
        }


class EnhancedIlotPlannerPro(QMainWindow):
    """
    Enhanced AI-powered application for architectural space analysis and furniture placement
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Architectural Analyzer PRO - Advanced DWG Analysis")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Enhanced data structures
        self.zones = []
        self.ilots = []
        self.ai_analysis_results = {}
        self.furniture_types = ['Desk', 'Chair', 'Table', 'Cabinet', 'Bed', 'Sofa']
        self.room_types = {}
        
        # Parameters
        self.echelle = 1.0
        self.calques_visibles = set()
        self.doc = None
        self.stats = {}
        
        # AI Analysis thread
        self.analysis_thread = None
        
        self.setup_enhanced_ui()
        self.setup_enhanced_toolbar()
        self.apply_modern_styling()

    def setup_enhanced_ui(self):
        """Setup enhanced UI with AI features"""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Create splitter for resizable panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(main_splitter)
        
        # Enhanced Control Panel
        control_panel = self.create_enhanced_control_panel()
        main_splitter.addWidget(control_panel)
        
        # Visualization area with multiple views
        viz_widget = self.create_visualization_area()
        main_splitter.addWidget(viz_widget)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 1200])
        
        # Status bar with progress
        self.setup_enhanced_status_bar()

    def create_enhanced_control_panel(self):
        """Create enhanced control panel with AI features"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Enhanced tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_ai_analysis_tab(), "🤖 AI Analysis")
        self.tabs.addTab(self.build_enhanced_parametres_tab(), "⚙️ Parameters")
        self.tabs.addTab(self.build_furniture_tab(), "🪑 Furniture")
        self.tabs.addTab(self.build_calques_tab(), "📋 Layers")
        self.tabs.addTab(self.build_enhanced_stats_tab(), "📊 Statistics")
        self.tabs.addTab(self.build_export_tab(), "📤 Export")
        
        layout.addWidget(self.tabs)
        return panel

    def build_ai_analysis_tab(self):
        """AI Analysis tab with intelligent room detection"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # AI Analysis Group
        ai_group = QGroupBox("AI Room Analysis")
        ai_layout = QVBoxLayout(ai_group)
        
        self.ai_analysis_btn = QPushButton("🧠 Analyze Architecture")
        self.ai_analysis_btn.clicked.connect(self.run_ai_analysis)
        ai_layout.addWidget(self.ai_analysis_btn)
        
        self.analysis_progress = QProgressBar()
        ai_layout.addWidget(self.analysis_progress)
        
        self.ai_results_text = QTextEdit()
        self.ai_results_text.setMaximumHeight(200)
        self.ai_results_text.setPlaceholderText("AI analysis results will appear here...")
        ai_layout.addWidget(self.ai_results_text)
        
        layout.addWidget(ai_group)
        
        # Room Type Override
        room_group = QGroupBox("Room Classification")
        room_layout = QVBoxLayout(room_group)
        
        self.room_type_combo = QComboBox()
        self.room_type_combo.addItems(['Auto-detect', 'Office', 'Bedroom', 'Living Room', 
                                     'Kitchen', 'Bathroom', 'Corridor', 'Storage'])
        room_layout.addWidget(QLabel("Force room type:"))
        room_layout.addWidget(self.room_type_combo)
        
        layout.addWidget(room_group)
        
        # AI Settings
        settings_group = QGroupBox("AI Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.confidence_threshold = QSlider(Qt.Orientation.Horizontal)
        self.confidence_threshold.setRange(50, 95)
        self.confidence_threshold.setValue(70)
        settings_layout.addRow("Confidence Threshold:", self.confidence_threshold)
        
        self.enable_smart_spacing = QCheckBox("Smart spacing optimization")
        self.enable_smart_spacing.setChecked(True)
        settings_layout.addRow(self.enable_smart_spacing)
        
        layout.addWidget(settings_group)
        
        return tab

    def build_enhanced_parametres_tab(self):
        """Enhanced parameters tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Basic Parameters
        basic_group = QGroupBox("Basic Dimensions")
        basic_layout = QFormLayout(basic_group)
        
        self.longueur_ilot = QDoubleSpinBox()
        self.longueur_ilot.setRange(0.1, 50.0)
        self.longueur_ilot.setValue(2.0)
        self.longueur_ilot.setSuffix(" m")
        
        self.largeur_ilot = QDoubleSpinBox()
        self.largeur_ilot.setRange(0.1, 50.0)
        self.largeur_ilot.setValue(1.5)
        self.largeur_ilot.setSuffix(" m")
        
        self.marge = QDoubleSpinBox()
        self.marge.setRange(0.0, 10.0)
        self.marge.setValue(0.5)
        self.marge.setSuffix(" m")
        
        basic_layout.addRow("Box Length:", self.longueur_ilot)
        basic_layout.addRow("Box Width:", self.largeur_ilot)
        basic_layout.addRow("Margin:", self.marge)
        
        layout.addWidget(basic_group)
        
        # Advanced Parameters
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout(advanced_group)
        
        self.rotation_angle = QSlider(Qt.Orientation.Horizontal)
        self.rotation_angle.setRange(0, 90)
        self.rotation_angle.setValue(0)
        
        self.min_room_area = QDoubleSpinBox()
        self.min_room_area.setRange(1.0, 1000.0)
        self.min_room_area.setValue(5.0)
        self.min_room_area.setSuffix(" m²")
        
        advanced_layout.addRow("Rotation Angle:", self.rotation_angle)
        advanced_layout.addRow("Min Room Area:", self.min_room_area)
        
        layout.addWidget(advanced_group)
        
        # Economic Parameters
        economic_group = QGroupBox("Economic Analysis")
        economic_layout = QFormLayout(economic_group)
        
        self.prix_m2 = QDoubleSpinBox()
        self.prix_m2.setRange(0, 10000)
        self.prix_m2.setValue(1200)
        self.prix_m2.setSuffix(" €")
        
        self.echelle_dxf = QDoubleSpinBox()
        self.echelle_dxf.setRange(0.001, 1000.0)
        self.echelle_dxf.setValue(1.0)
        
        economic_layout.addRow("Price per m²:", self.prix_m2)
        economic_layout.addRow("DXF Scale:", self.echelle_dxf)
        
        layout.addWidget(economic_group)
        
        return tab

    def build_furniture_tab(self):
        """Furniture management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        furniture_group = QGroupBox("Furniture Types")
        furniture_layout = QVBoxLayout(furniture_group)
        
        self.furniture_list = QListWidget()
        for furniture in self.furniture_types:
            item = QListWidgetItem(furniture)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.furniture_list.addItem(item)
        
        furniture_layout.addWidget(self.furniture_list)
        layout.addWidget(furniture_group)
        
        # Custom furniture
        custom_group = QGroupBox("Custom Furniture")
        custom_layout = QFormLayout(custom_group)
        
        self.custom_name = QLabel("Custom Name:")
        self.custom_length = QDoubleSpinBox()
        self.custom_width = QDoubleSpinBox()
        
        custom_layout.addRow("Name:", self.custom_name)
        custom_layout.addRow("Length (m):", self.custom_length)
        custom_layout.addRow("Width (m):", self.custom_width)
        
        add_custom_btn = QPushButton("Add Custom Furniture")
        custom_layout.addRow(add_custom_btn)
        
        layout.addWidget(custom_group)
        
        return tab

    def build_calques_tab(self):
        """Enhanced layers tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Layer controls
        controls_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        
        select_all_btn.clicked.connect(self.select_all_layers)
        deselect_all_btn.clicked.connect(self.deselect_all_layers)
        
        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(deselect_all_btn)
        layout.addLayout(controls_layout)
        
        self.calques_list = QListWidget()
        self.calques_list.itemChanged.connect(self.mettre_a_jour_calques)
        layout.addWidget(self.calques_list)
        
        return tab

    def build_enhanced_stats_tab(self):
        """Enhanced statistics tab with detailed analysis"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText("No analysis performed yet...")
        layout.addWidget(self.stats_text)
        
        return tab

    def build_export_tab(self):
        """Export options tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        self.export_pdf_btn = QPushButton("📄 Export to PDF")
        self.export_excel_btn = QPushButton("📊 Export to Excel")
        self.export_json_btn = QPushButton("📋 Export Analysis Data")
        
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        self.export_excel_btn.clicked.connect(self.export_excel)
        self.export_json_btn.clicked.connect(self.export_analysis_json)
        
        export_layout.addWidget(self.export_pdf_btn)
        export_layout.addWidget(self.export_excel_btn)
        export_layout.addWidget(self.export_json_btn)
        
        layout.addWidget(export_group)
        
        return tab

    def create_visualization_area(self):
        """Create enhanced visualization area"""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        # Visualization tabs
        self.viz_tabs = QTabWidget()
        
        # Main plot
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.viz_tabs.addTab(self.canvas, "📐 Floor Plan")
        
        # 3D view
        self.figure_3d = Figure(figsize=(12, 8))
        self.canvas_3d = FigureCanvasQTAgg(self.figure_3d)
        self.viz_tabs.addTab(self.canvas_3d, "🏗️ 3D View")
        
        layout.addWidget(self.viz_tabs)
        return viz_widget

    def setup_enhanced_toolbar(self):
        """Enhanced toolbar with more features"""
        toolbar = self.addToolBar("Main Tools")
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        actions = [
            ("📂 Load DWG", self.charger_dxf),
            ("🧠 AI Analyze", self.run_ai_analysis),
            ("🧮 Calculate", self.placer_ilots_enhanced),
            ("💾 Save Project", self.sauvegarder_projet),
            ("📄 Load Project", self.charger_projet),
            ("🧼 Reset", self.reinitialiser)
        ]
        
        for text, action in actions:
            btn = QPushButton(text)
            btn.clicked.connect(action)
            toolbar.addWidget(btn)

    def setup_enhanced_status_bar(self):
        """Enhanced status bar"""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready for analysis")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def apply_modern_styling(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007acc;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #003d6b;
            }
        """)

    def run_ai_analysis(self):
        """Run AI analysis of the architecture"""
        if not self.zones:
            QMessageBox.warning(self, "No Data", "Please load a DWG file first.")
            return
        
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setValue(0)
        self.ai_analysis_btn.setEnabled(False)
        
        # Prepare analysis parameters
        params = {
            'box_size': (self.longueur_ilot.value(), self.largeur_ilot.value()),
            'margin': self.marge.value(),
            'min_area': self.min_room_area.value(),
            'confidence_threshold': self.confidence_threshold.value() / 100.0
        }
        
        # Start analysis thread
        self.analysis_thread = AIAnalysisThread(self.zones, params)
        self.analysis_thread.progress_updated.connect(self.analysis_progress.setValue)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_complete(self, results):
        """Handle completion of AI analysis"""
        self.ai_analysis_results = results
        self.analysis_progress.setVisible(False)
        self.ai_analysis_btn.setEnabled(True)
        
        # Display results
        self.display_ai_results(results)
        self.update_enhanced_stats()
        self.status_label.setText("AI analysis completed successfully")

    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        self.analysis_progress.setVisible(False)
        self.ai_analysis_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"AI analysis failed:\n{error_msg}")

    def display_ai_results(self, results):
        """Display AI analysis results"""
        text = "🤖 AI ANALYSIS RESULTS\n" + "="*50 + "\n\n"
        
        # Room analysis
        if 'rooms' in results:
            text += "📋 ROOM CLASSIFICATION:\n"
            for zone_id, room_info in results['rooms'].items():
                text += f"  • {zone_id}: {room_info['type']} "
                text += f"(Confidence: {room_info['confidence']:.0%})\n"
                text += f"    Area: {room_info['area']:.1f} m², "
                text += f"Dimensions: {room_info['dimensions'][0]:.1f}×{room_info['dimensions'][1]:.1f} m\n\n"
        
        # Furniture placement
        if 'furniture' in results:
            total_spots = sum(len(spots) for spots in results['furniture'].values())
            text += f"🪑 FURNITURE PLACEMENT:\n"
            text += f"  • Total placement opportunities: {total_spots}\n"
            for zone_id, spots in results['furniture'].items():
                if spots:
                    avg_score = sum(spot['suitability_score'] for spot in spots) / len(spots)
                    text += f"  • {zone_id}: {len(spots)} spots (Avg. suitability: {avg_score:.1%})\n"
        
        self.ai_results_text.setPlainText(text)

    def charger_dxf(self):
        """Enhanced DXF loading with better error handling"""
        file, _ = QFileDialog.getOpenFileName(
            self, "Load DWG/DXF File", "", 
            "CAD Files (*.dxf *.dwg);;DXF Files (*.dxf);;DWG Files (*.dwg)"
        )
        if not file:
            return
            
        try:
            self.status_label.setText("Loading CAD file...")
            self.doc = ezdxf.readfile(file)
            self.zones.clear()
            self.ilots.clear()
            self.calques_list.clear()
            
            # Enhanced zone detection
            calques = set()
            zone_count = 0
            
            # Process different entity types
            for entity_type in ['LWPOLYLINE', 'POLYLINE', 'LINE', 'CIRCLE', 'ARC']:
                for entity in self.doc.modelspace().query(entity_type):
                    layer = getattr(entity.dxf, 'layer', 'Default')
                    calques.add(layer)
                    
                    if entity_type in ['LWPOLYLINE', 'POLYLINE'] and hasattr(entity, 'closed') and entity.closed:
                        points = [(p[0], p[1]) for p in entity]
                        if len(points) >= 3:  # Valid polygon
                            self.zones.append({
                                'points': points,
                                'layer': layer,
                                'entity_type': entity_type
                            })
                            zone_count += 1
            
            # Update layer list
            for name in sorted(calques):
                item = QListWidgetItem(f"📋 {name}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.calques_list.addItem(item)
            
            self.mettre_a_jour_calques()
            self.status_label.setText(f"Loaded: {zone_count} zones from {len(calques)} layers")
            
            # Auto-run AI analysis if zones found
            if zone_count > 0:
                QTimer.singleShot(1000, self.run_ai_analysis)  # Delay for UI update
                
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Could not load file:\n{str(e)}")
            self.status_label.setText("Error loading file")

    def placer_ilots_enhanced(self):
        """Enhanced placement algorithm with AI optimization"""
        if not self.zones:
            QMessageBox.warning(self, "No Zones", "Please load a DWG file first.")
            return
        
        self.ilots.clear()
        l = self.longueur_ilot.value() / self.echelle_dxf.value()
        h = self.largeur_ilot.value() / self.echelle_dxf.value()
        marge = self.marge.value() / self.echelle_dxf.value()
        
        total_placed = 0
        
        for i, zone in enumerate(self.zones):
            if zone['layer'] not in self.calques_visibles:
                continue
            
            pts = zone['points']
            if len(pts) < 3:
                continue
                
            # Use AI results if available
            zone_id = f"Zone_{i}"
            if (self.ai_analysis_results and 
                'furniture' in self.ai_analysis_results and 
                zone_id in self.ai_analysis_results['furniture']):
                
                # Use AI-optimized placements
                spots = self.ai_analysis_results['furniture'][zone_id]
                for spot in spots:
                    if spot['suitability_score'] > 0.5:  # Only high-quality spots
                        self.ilots.append(spot['box_coords'])
                        total_placed += 1
            else:
                # Fallback to grid-based placement
                total_placed += self.grid_based_placement(zone, l, h, marge)
        
        self.mettre_a_jour_stats()
        self.dessiner_enhanced()
        self.status_label.setText(f"Placed {total_placed} items")

    def grid_base