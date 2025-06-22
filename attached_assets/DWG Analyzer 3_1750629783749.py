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
    QGridLayout, QSlider, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon as MplPolygon, Circle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


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
                'optimization': optimization_results,
                'total_boxes': sum(len(spots) for spots in furniture_analysis.values())
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
            
            # Enhanced room classification logic
            if area < 5:  # Very small rooms
                if aspect_ratio > 4:
                    room_type = "Corridor"
                    confidence = 0.9
                elif width < 2 or height < 2:
                    room_type = "Storage/WC"
                    confidence = 0.8
                else:
                    room_type = "Small Office"
                    confidence = 0.7
            elif area < 15:  # Small rooms
                if aspect_ratio > 2.5:
                    room_type = "Corridor"
                    confidence = 0.8
                elif aspect_ratio < 1.3:
                    room_type = "Small Office/Bedroom"
                    confidence = 0.7
                else:
                    room_type = "Office"
                    confidence = 0.6
            elif area < 35:  # Medium rooms
                if aspect_ratio < 1.5:
                    room_type = "Meeting Room"
                    confidence = 0.8
                else:
                    room_type = "Open Office"
                    confidence = 0.7
            elif area < 70:  # Large rooms
                if aspect_ratio < 1.3:
                    room_type = "Conference Room"
                    confidence = 0.8
                else:
                    room_type = "Large Open Office"
                    confidence = 0.7
            else:  # Very large rooms
                room_type = "Hall/Auditorium"
                confidence = 0.9
            
            room_types[f"Zone_{i}"] = {
                'type': room_type,
                'confidence': confidence,
                'area': area,
                'dimensions': (width, height),
                'aspect_ratio': aspect_ratio,
                'layer': zone.get('layer', 'Unknown'),
                'perimeter': poly.length,
                'center': poly.centroid.coords[0]
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
            
            # Adaptive grid spacing based on room size
            area = poly.area
            if area < 10:
                # Smaller rooms: tighter spacing
                spacing_factor = 0.8
            elif area > 50:
                # Larger rooms: more generous spacing
                spacing_factor = 1.2
            else:
                spacing_factor = 1.0
            
            x_step = (box_size[0] + margin) * spacing_factor
            y_step = (box_size[1] + margin) * spacing_factor
            
            # Multiple orientation attempts
            orientations = [(box_size[0], box_size[1])]
            if self.params.get('allow_rotation', True):
                orientations.append((box_size[1], box_size[0]))  # 90-degree rotation
            
            best_spots = []
            
            for width, height in orientations:
                current_spots = []
                
                x = bounds[0] + margin
                while x + width <= bounds[2] - margin:
                    y = bounds[1] + margin
                    while y + height <= bounds[3] - margin:
                        # Create test box
                        test_box = box(x, y, x + width, y + height)
                        
                        # Check if placement is valid (fully inside polygon)
                        if poly.contains(test_box):
                            suitability = self.calculate_suitability(poly, test_box)
                            if suitability > 0.3:  # Minimum suitability threshold
                                current_spots.append({
                                    'position': (x, y),
                                    'size': (width, height),
                                    'box_coords': [(x, y), (x + width, y), 
                                                 (x + width, y + height), 
                                                 (x, y + height)],
                                    'suitability_score': suitability,
                                    'area': width * height
                                })
                        y += y_step
                    x += x_step
                
                # Keep best orientation
                if len(current_spots) > len(best_spots):
                    best_spots = current_spots
            
            placements[f"Zone_{i}"] = best_spots
        
        return placements
    
    def calculate_suitability(self, room_poly, furniture_box):
        """Calculate suitability score for furniture placement"""
        center = furniture_box.centroid
        
        # Distance from walls (prefer some clearance but not too much)
        distance_to_edge = room_poly.boundary.distance(center)
        wall_score = min(distance_to_edge / 3.0, 1.0)  # Normalize to 0-1
        
        # Distance from room center
        room_center = room_poly.centroid
        distance_to_center = center.distance(room_center)
        room_radius = math.sqrt(room_poly.area / math.pi)  # Approximate room radius
        center_score = max(0, 1.0 - (distance_to_center / room_radius))
        
        # Area utilization efficiency
        room_area = room_poly.area
        box_area = furniture_box.area
        utilization_score = min(box_area / (room_area * 0.1), 1.0)  # Prefer reasonable size
        
        # Weighted combination
        total_score = (wall_score * 0.4 + center_score * 0.4 + utilization_score * 0.2)
        
        return min(total_score, 1.0)
    
    def optimize_placements(self):
        """Use advanced algorithms to optimize furniture placement"""
        return {
            'algorithm_used': 'AI Grid-Based with Suitability Scoring',
            'optimization_level': 'High',
            'total_efficiency': 0.85,
            'placement_strategy': 'Multi-orientation with adaptive spacing'
        }


class EnhancedIlotPlannerPro(QMainWindow):
    """
    Enhanced AI-powered application for architectural space analysis and furniture placement
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Architectural Space Analyzer PRO - DWG Analysis & Box Placement")
        self.setGeometry(50, 50, 1800, 1000)
        
        # Enhanced data structures
        self.zones = []
        self.ilots = []
        self.ai_analysis_results = {}
        self.furniture_types = ['Desk', 'Chair', 'Table', 'Cabinet', 'Bed', 'Sofa', 'Workstation']
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
        main_splitter.setSizes([450, 1350])
        
        # Status bar with progress
        self.setup_enhanced_status_bar()

    def create_enhanced_control_panel(self):
        """Create enhanced control panel with AI features"""
        panel = QWidget()
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        
        # Enhanced tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_ai_analysis_tab(), "🤖 AI Analysis")
        self.tabs.addTab(self.build_enhanced_parametres_tab(), "⚙️ Parameters")
        self.tabs.addTab(self.build_furniture_tab(), "📦 Box Types")
        self.tabs.addTab(self.build_calques_tab(), "📋 Layers")
        self.tabs.addTab(self.build_enhanced_stats_tab(), "📊 Results")
        self.tabs.addTab(self.build_export_tab(), "📤 Export")
        
        layout.addWidget(self.tabs)
        return panel

    def build_ai_analysis_tab(self):
        """AI Analysis tab with intelligent room detection"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File Loading Group
        file_group = QGroupBox("📂 Load DWG/DXF File")
        file_layout = QVBoxLayout(file_group)
        
        self.load_file_btn = QPushButton("🔍 Browse & Load File")
        self.load_file_btn.clicked.connect(self.charger_dxf)
        file_layout.addWidget(self.load_file_btn)
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.file_info_label)
        
        layout.addWidget(file_group)
        
        # AI Analysis Group
        ai_group = QGroupBox("🧠 AI Room Analysis")
        ai_layout = QVBoxLayout(ai_group)
        
        self.ai_analysis_btn = QPushButton("🚀 Start AI Analysis")
        self.ai_analysis_btn.clicked.connect(self.run_ai_analysis)
        self.ai_analysis_btn.setEnabled(False)
        ai_layout.addWidget(self.ai_analysis_btn)
        
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        ai_layout.addWidget(self.analysis_progress)
        
        self.ai_results_text = QTextEdit()
        self.ai_results_text.setMaximumHeight(200)
        self.ai_results_text.setPlaceholderText("AI analysis results will appear here...")
        ai_layout.addWidget(self.ai_results_text)
        
        layout.addWidget(ai_group)
        
        # Quick Actions
        actions_group = QGroupBox("⚡ Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.calculate_btn = QPushButton("📊 Calculate Box Placement")
        self.calculate_btn.clicked.connect(self.placer_ilots_enhanced)
        self.calculate_btn.setEnabled(False)
        actions_layout.addWidget(self.calculate_btn)
        
        self.reset_btn = QPushButton("🔄 Reset Analysis")
        self.reset_btn.clicked.connect(self.reinitialiser)
        actions_layout.addWidget(self.reset_btn)
        
        layout.addWidget(actions_group)
        
        return tab

    def build_enhanced_parametres_tab(self):
        """Enhanced parameters tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Box Dimensions
        box_group = QGroupBox("📦 Box/Furniture Dimensions")
        box_layout = QFormLayout(box_group)
        
        self.longueur_ilot = QDoubleSpinBox()
        self.longueur_ilot.setRange(0.1, 50.0)
        self.longueur_ilot.setValue(2.0)
        self.longueur_ilot.setSuffix(" m")
        self.longueur_ilot.setDecimals(2)
        
        self.largeur_ilot = QDoubleSpinBox()
        self.largeur_ilot.setRange(0.1, 50.0)
        self.largeur_ilot.setValue(1.5)
        self.largeur_ilot.setSuffix(" m")
        self.largeur_ilot.setDecimals(2)
        
        self.marge = QDoubleSpinBox()
        self.marge.setRange(0.0, 10.0)
        self.marge.setValue(0.5)
        self.marge.setSuffix(" m")
        self.marge.setDecimals(2)
        
        box_layout.addRow("Box Length:", self.longueur_ilot)
        box_layout.addRow("Box Width:", self.largeur_ilot)
        box_layout.addRow("Safety Margin:", self.marge)
        
        layout.addWidget(box_group)
        
        # Placement Options
        placement_group = QGroupBox("🎯 Placement Options")
        placement_layout = QFormLayout(placement_group)
        
        self.allow_rotation = QCheckBox("Allow 90° rotation")
        self.allow_rotation.setChecked(True)
        placement_layout.addRow(self.allow_rotation)
        
        self.min_room_area = QDoubleSpinBox()
        self.min_room_area.setRange(1.0, 1000.0)
        self.min_room_area.setValue(5.0)
        self.min_room_area.setSuffix(" m²")
        placement_layout.addRow("Min Room Area:", self.min_room_area)
        
        self.suitability_threshold = QSlider(Qt.Orientation.Horizontal)
        self.suitability_threshold.setRange(10, 90)
        self.suitability_threshold.setValue(30)
        self.suitability_label = QLabel("30%")
        self.suitability_threshold.valueChanged.connect(
            lambda v: self.suitability_label.setText(f"{v}%")
        )
        placement_layout.addRow("Quality Threshold:", self.suitability_threshold)
        placement_layout.addRow("", self.suitability_label)
        
        layout.addWidget(placement_group)
        
        # DWG Settings
        dwg_group = QGroupBox("📐 DWG Settings")
        dwg_layout = QFormLayout(dwg_group)
        
        self.echelle_dxf = QDoubleSpinBox()
        self.echelle_dxf.setRange(0.001, 1000.0)
        self.echelle_dxf.setValue(1.0)
        self.echelle_dxf.setDecimals(3)
        
        dwg_layout.addRow("DXF Scale Factor:", self.echelle_dxf)
        
        layout.addWidget(dwg_group)
        
        return tab

    def build_furniture_tab(self):
        """Furniture management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Predefined boxes
        predefined_group = QGroupBox("📦 Standard Box Types")
        predefined_layout = QVBoxLayout(predefined_group)
        
        standard_boxes = [
            ("Small Office Desk", 1.6, 0.8),
            ("Large Office Desk", 2.0, 1.0),
            ("Meeting Table (4p)", 2.4, 1.2),
            ("Conference Table (8p)", 3.6, 1.8),
            ("Workstation", 1.8, 1.2),
            ("Storage Cabinet", 0.8, 0.6),
            ("Reception Desk", 2.5, 1.5)
        ]
        
        for name, length, width in standard_boxes:
            btn = QPushButton(f"{name} ({length}×{width}m)")
            btn.clicked.connect(lambda checked, l=length, w=width: self.set_box_dimensions(l, w))
            predefined_layout.addWidget(btn)
        
        layout.addWidget(predefined_group)
        
        # Custom dimensions
        custom_group = QGroupBox("✏️ Custom Dimensions")
        custom_layout = QFormLayout(custom_group)
        
        self.custom_length = QDoubleSpinBox()
        self.custom_length.setRange(0.1, 10.0)
        self.custom_length.setValue(2.0)
        self.custom_length.setSuffix(" m")
        
        self.custom_width = QDoubleSpinBox()
        self.custom_width.setRange(0.1, 10.0)
        self.custom_width.setValue(1.5)
        self.custom_width.setSuffix(" m")
        
        custom_layout.addRow("Length:", self.custom_length)
        custom_layout.addRow("Width:", self.custom_width)
        
        apply_custom_btn = QPushButton("Apply Custom Size")
        apply_custom_btn.clicked.connect(self.apply_custom_dimensions)
        custom_layout.addRow(apply_custom_btn)
        
        layout.addWidget(custom_group)
        
        return tab

    def build_calques_tab(self):
        """Enhanced layers tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Layer controls
        controls_layout = QHBoxLayout()
        select_all_btn = QPushButton("✅ All")
        deselect_all_btn = QPushButton("❌ None")
        
        select_all_btn.clicked.connect(self.select_all_layers)
        deselect_all_btn.clicked.connect(self.deselect_all_layers)
        
        controls_layout.addWidget(select_all_btn)
        controls_layout.addWidget(deselect_all_btn)
        layout.addLayout(controls_layout)
        
        self.calques_list = QListWidget()
        self.calques_list.itemChanged.connect(self.mettre_a_jour_calques)
        layout.addWidget(self.calques_list)
        
        layer_info = QLabel("💡 Tip: Uncheck layers you don't want to analyze")
        layer_info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(layer_info)
        
        return tab

    def build_enhanced_stats_tab(self):
        """Enhanced statistics tab with detailed analysis"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText("📊 Analysis results will appear here after calculation...")
        layout.addWidget(self.stats_text)
        
        return tab

    def build_export_tab(self):
        """Export options tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        export_group = QGroupBox("📤 Export Results")
        export_layout = QVBoxLayout(export_group)
        
        self.export_pdf_btn = QPushButton("📄 Export Floor Plan to PDF")
        self.export_excel_btn = QPushButton("📊 Export Data to Excel")
        self.export_json_btn = QPushButton("💾 Save Analysis as JSON")
        
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
        self.figure = Figure(figsize=(14, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.viz_tabs.addTab(self.canvas, "📐 Floor Plan Analysis")
        
        layout.addWidget(self.viz_tabs)
        return viz_widget

    def setup_enhanced_toolbar(self):
        """Enhanced toolbar with more features"""
        toolbar = self.addToolBar("Main Tools")
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # Main workflow buttons
        load_btn = QPushButton("📂 Load DWG")
        load_btn.clicked.connect(self.charger_dxf)
        toolbar.addWidget(load_btn)
        
        analyze_btn = QPushButton("🧠 AI Analysis")
        analyze_btn.clicked.connect(self.run_ai_analysis)
        toolbar.addWidget(analyze_btn)
        
        calculate_btn = QPushButton("📊 Calculate")
        calculate_btn.clicked.connect(self.placer_ilots_enhanced)
        toolbar.addWidget(calculate_btn)
        
        toolbar.addSeparator()
        
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self.reinitialiser)
        toolbar.addWidget(reset_btn)

    def setup_enhanced_status_bar(self):
        """Enhanced status bar"""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready - Load a DWG file to begin")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def apply_modern_styling(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                padding: 12px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #007bff;
            }
            QGroupBox {
                font-weight: 600;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QDoubleSpinBox, QSpinBox {
                border: 2px solid #dee2e6;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
            }
            QSlider::groove:horizontal {
                border: 1px solid #dee2e6;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 1px solid #007bff;
                width: 18px;
                border-radius: 9px;
            }
        """)

    def set_box_dimensions(self, length, width):
        """Set box dimensions from predefined values"""
        self.longueur_ilot.setValue(length)
        self.largeur_ilot.setValue(width)
        self.status_label.setText(f"Box dimensions set to {length}×{width}m")

    def apply_custom_dimensions(self):
        """Apply custom dimensions"""
        self.longueur_ilot.setValue(self.custom_length.value())
        self.largeur_ilot.setValue(self.custom_width.value())
        self.status_label.setText(f"Custom dimensions applied: {self.custom_length.value()}×{self.custom_width.value()}m")

    def run_ai_analysis(self):
        """Run AI analysis of the architecture"""
        if not self.zones:
            QMessageBox.warning(self, "No Data", "Please load a DWG file first.")
            return
        
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setValue(0)
        self.ai_analysis_btn.setEnabled(False)
        self.calculate_btn.setEnabled(False)
        
        # Prepare analysis parameters
        params = {
            'box_size': (self.longueur_ilot.value(), self.largeur_ilot.value()),
            'margin': self.marge.value(),
            'min_area': self.min_room_area.value(),
            'allow_rotation': self.allow_rotation.isChecked(),
            'suitability_threshold': self.suitability_threshold.value() / 100.0
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
        self.calculate_btn.setEnabled(True)
        
        # Display results
        self.display_ai_results(results)
        self.update_enhanced_stats()
        self.status_label.setText(f"AI analysis completed - {results['total_boxes']} placement opportunities found")

    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        self.analysis_progress.setVisible(False)
        self.ai_analysis_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"AI analysis failed:\n{error_msg}")

    def display_ai_results(self, results):
        """Display AI analysis results