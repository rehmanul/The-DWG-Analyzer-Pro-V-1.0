import io
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

class ExportManager:
    """
    Utility class for exporting analysis results in various formats
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        )
    
    def generate_pdf_report(self, zones: List[Dict], analysis_results: Dict) -> bytes:
        """
        Generate comprehensive PDF report
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("AI Architectural Space Analysis Report", self.title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              self.styles['Normal']))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(zones, analysis_results))
        story.append(PageBreak())
        
        # Detailed room analysis
        story.extend(self._create_room_analysis_section(analysis_results))
        story.append(PageBreak())
        
        # Box placement analysis
        story.extend(self._create_placement_analysis_section(analysis_results))
        story.append(PageBreak())
        
        # Statistical summary
        story.extend(self._create_statistical_summary(analysis_results))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_executive_summary(self, zones: List[Dict], analysis_results: Dict) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        total_zones = len(zones)
        total_boxes = analysis_results.get('total_boxes', 0)
        
        if analysis_results.get('parameters'):
            params = analysis_results['parameters']
            box_area = params['box_size'][0] * params['box_size'][1]
            total_utilized_area = total_boxes * box_area
        else:
            total_utilized_area = 0
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Zones Analyzed', str(total_zones)],
            ['Total Box Placements', str(total_boxes)],
            ['Total Utilized Area', f"{total_utilized_area:.1f} m²"],
        ]
        
        if analysis_results.get('rooms'):
            room_types = list(set(info['type'] for info in analysis_results['rooms'].values()))
            summary_data.append(['Room Types Found', ', '.join(room_types)])
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Key findings
        story.append(Paragraph("Key Findings", self.styles['Heading3']))
        findings = []
        
        if analysis_results.get('rooms'):
            # Most common room type
            room_types = [info['type'] for info in analysis_results['rooms'].values()]
            most_common = max(set(room_types), key=room_types.count) if room_types else "None"
            findings.append(f"Most common room type: {most_common}")
            
            # Average confidence
            confidences = [info['confidence'] for info in analysis_results['rooms'].values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            findings.append(f"Average room classification confidence: {avg_confidence:.1%}")
        
        if analysis_results.get('optimization'):
            efficiency = analysis_results['optimization'].get('total_efficiency', 0)
            findings.append(f"Overall placement efficiency: {efficiency:.1%}")
        
        for finding in findings:
            story.append(Paragraph(f"• {finding}", self.styles['Normal']))
        
        return story
    
    def _create_room_analysis_section(self, analysis_results: Dict) -> List:
        """Create room analysis section"""
        story = []
        
        story.append(Paragraph("Detailed Room Analysis", self.heading_style))
        
        if not analysis_results.get('rooms'):
            story.append(Paragraph("No room analysis data available.", self.styles['Normal']))
            return story
        
        # Create room analysis table
        room_data = [['Zone', 'Type', 'Confidence', 'Area (m²)', 'Dimensions (m)', 'Layer']]
        
        for zone_name, room_info in analysis_results['rooms'].items():
            room_data.append([
                zone_name,
                room_info['type'],
                f"{room_info['confidence']:.1%}",
                f"{room_info['area']:.1f}",
                f"{room_info['dimensions'][0]:.1f} × {room_info['dimensions'][1]:.1f}",
                room_info.get('layer', 'Unknown')
            ])
        
        room_table = Table(room_data)
        room_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(room_table)
        
        return story
    
    def _create_placement_analysis_section(self, analysis_results: Dict) -> List:
        """Create box placement analysis section"""
        story = []
        
        story.append(Paragraph("Box Placement Analysis", self.heading_style))
        
        if not analysis_results.get('placements'):
            story.append(Paragraph("No placement analysis data available.", self.styles['Normal']))
            return story
        
        # Summary by zone
        placement_summary = [['Zone', 'Boxes Placed', 'Total Area (m²)', 'Avg. Suitability']]
        
        for zone_name, placements in analysis_results['placements'].items():
            if placements:
                total_area = sum(p['area'] for p in placements)
                avg_suitability = sum(p['suitability_score'] for p in placements) / len(placements)
                
                placement_summary.append([
                    zone_name,
                    str(len(placements)),
                    f"{total_area:.1f}",
                    f"{avg_suitability:.2f}"
                ])
        
        placement_table = Table(placement_summary)
        placement_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(placement_table)
        story.append(Spacer(1, 20))
        
        # Parameters used
        if analysis_results.get('parameters'):
            story.append(Paragraph("Placement Parameters", self.styles['Heading3']))
            params = analysis_results['parameters']
            
            param_data = [
                ['Parameter', 'Value'],
                ['Box Size', f"{params['box_size'][0]} × {params['box_size'][1]} m"],
                ['Margin', f"{params['margin']} m"],
                ['Allow Rotation', 'Yes' if params.get('allow_rotation', False) else 'No'],
                ['Smart Spacing', 'Yes' if params.get('smart_spacing', False) else 'No']
            ]
            
            param_table = Table(param_data)
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(param_table)
        
        return story
    
    def _create_statistical_summary(self, analysis_results: Dict) -> List:
        """Create statistical summary section"""
        story = []
        
        story.append(Paragraph("Statistical Summary", self.heading_style))
        
        # Calculate statistics
        if analysis_results.get('rooms') and analysis_results.get('placements'):
            rooms = analysis_results['rooms']
            placements = analysis_results['placements']
            
            # Room type statistics
            room_types = [info['type'] for info in rooms.values()]
            room_type_counts = {}
            for room_type in room_types:
                room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            
            story.append(Paragraph("Room Type Distribution:", self.styles['Heading3']))
            for room_type, count in sorted(room_type_counts.items()):
                percentage = (count / len(room_types)) * 100
                story.append(Paragraph(f"• {room_type}: {count} rooms ({percentage:.1f}%)", 
                                     self.styles['Normal']))
            
            story.append(Spacer(1, 12))
            
            # Efficiency statistics
            story.append(Paragraph("Efficiency Metrics:", self.styles['Heading3']))
            
            total_room_area = sum(info['area'] for info in rooms.values())
            total_box_area = sum(sum(p['area'] for p in zone_placements) 
                               for zone_placements in placements.values())
            
            if total_room_area > 0:
                utilization = (total_box_area / total_room_area) * 100
                story.append(Paragraph(f"• Space Utilization: {utilization:.1f}%", 
                                     self.styles['Normal']))
            
            # Suitability statistics
            all_scores = []
            for zone_placements in placements.values():
                all_scores.extend([p['suitability_score'] for p in zone_placements])
            
            if all_scores:
                avg_suitability = sum(all_scores) / len(all_scores)
                min_suitability = min(all_scores)
                max_suitability = max(all_scores)
                
                story.append(Paragraph(f"• Average Suitability Score: {avg_suitability:.2f}", 
                                     self.styles['Normal']))
                story.append(Paragraph(f"• Suitability Range: {min_suitability:.2f} - {max_suitability:.2f}", 
                                     self.styles['Normal']))
        
        return story
    
    def export_to_json(self, analysis_results: Dict) -> str:
        """Export analysis results to JSON format"""
        # Create a clean copy of results for JSON export
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analysis_results': self._clean_for_json(analysis_results)
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def export_to_csv(self, analysis_results: Dict) -> str:
        """Export room analysis to CSV format"""
        if not analysis_results.get('rooms'):
            return "No room data available for export"
        
        # Prepare data for CSV
        csv_data = []
        rooms = analysis_results['rooms']
        placements = analysis_results.get('placements', {})
        
        for zone_name, room_info in rooms.items():
            zone_placements = placements.get(zone_name, [])
            csv_data.append({
                'Zone': zone_name,
                'Room_Type': room_info['type'],
                'Confidence': room_info['confidence'],
                'Area_m2': room_info['area'],
                'Width_m': room_info['dimensions'][0],
                'Height_m': room_info['dimensions'][1],
                'Aspect_Ratio': room_info.get('aspect_ratio', 0),
                'Boxes_Placed': len(zone_placements),
                'Total_Box_Area_m2': sum(p['area'] for p in zone_placements),
                'Avg_Suitability': sum(p['suitability_score'] for p in zone_placements) / len(zone_placements) if zone_placements else 0,
                'Layer': room_info.get('layer', 'Unknown')
            })
        
        # Convert to DataFrame and then to CSV
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # NumPy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return str(obj)
        else:
            return obj
