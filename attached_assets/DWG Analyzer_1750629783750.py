import sys
import math
import json
import numpy as np
import ezdxf
from rectpack import newPacker
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QLabel, QFileDialog, QFormLayout, QDoubleSpinBox, QMessageBox,
    QHBoxLayout, QListWidget, QListWidgetItem, QTabWidget
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D

class IlotPlannerPro(QMainWindow):
    """
    Application principale pour le placement d'îlots dans des zones définies par des fichiers DXF.
    """

    def __init__(self):
        """
        Initialise l'application principale.
        """
        super().__init__()
        self.setWindowTitle("Îlot Planner PRO - Architecture Optimisée")
        self.setGeometry(100, 100, 1400, 900)

        self.zones = []
        self.ilots = []
        self.echelle = 1.0
        self.calques_visibles = set()
        self.doc = None
        self.stats = {}

        self.setup_ui()
        self.setup_toolbar()

    def setup_ui(self):
        """
        Initialise l'interface utilisateur.
        """
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # -- CONTROL PANEL --
        control_panel = QWidget()
        control_panel.setMaximumWidth(320)
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_parametres_tab(), "Paramètres")
        self.tabs.addTab(self.build_calques_tab(), "Calques")
        self.tabs.addTab(self.build_stats_tab(), "Statistiques")
        control_layout.addWidget(self.tabs)

        self.status_bar = QLabel("Prêt")
        self.status_bar.setStyleSheet("background: #eee; padding: 8px; border-top: 1px solid #ccc;")
        control_layout.addWidget(self.status_bar)

        # -- MATPLOTLIB CANVAS --
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

    def setup_toolbar(self):
        """
        Initialise la barre d'outils principale.
        """
        toolbar = self.addToolBar("Outils")
        for label, action in [
            ("📂 Ouvrir", self.charger_dxf),
            ("💾 Sauvegarder", self.sauvegarder_projet),
            ("📄 Charger projet", self.charger_projet),
            ("🧮 Calculer", self.placer_ilots),
            ("📊 Vue 3D", self.afficher_3d),
            ("📤 Export PDF", self.export_pdf),
            ("🧼 Réinitialiser", self.reinitialiser)
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(action)
            toolbar.addWidget(btn)

    def build_parametres_tab(self):
        """
        Crée l'onglet des paramètres.
        """
        tab = QWidget()
        layout = QFormLayout(tab)

        self.longueur_ilot = QDoubleSpinBox()
        self.longueur_ilot.setValue(2.0)
        self.largeur_ilot = QDoubleSpinBox()
        self.largeur_ilot.setValue(1.5)
        self.marge = QDoubleSpinBox()
        self.marge.setValue(0.5)
        self.prix_m2 = QDoubleSpinBox()
        self.prix_m2.setValue(1200)
        self.echelle_dxf = QDoubleSpinBox()
        self.echelle_dxf.setValue(1.0)

        for label, widget in [
            ("Longueur îlot (m)", self.longueur_ilot),
            ("Largeur îlot (m)", self.largeur_ilot),
            ("Marge (m)", self.marge),
            ("Prix au m² (€)", self.prix_m2),
            ("Échelle DXF", self.echelle_dxf)
        ]:
            layout.addRow(label, widget)

        return tab

    def build_calques_tab(self):
        """
        Crée l'onglet de gestion des calques.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.calques_list = QListWidget()
        layout.addWidget(self.calques_list)
        return tab

    def build_stats_tab(self):
        """
        Crée l'onglet des statistiques.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.stats_label = QLabel("Aucun calcul effectué.")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        return tab

    def charger_dxf(self):
        """
        Charge un fichier DXF.
        """
        file, _ = QFileDialog.getOpenFileName(self, "Charger DXF", "", "Fichiers DXF (*.dxf)")
        if not file:
            return
        try:
            self.doc = ezdxf.readfile(file)
            self.zones.clear()
            self.ilots.clear()
            self.calques_list.clear()

            calques = set()
            for entity in self.doc.modelspace().query('LWPOLYLINE'):
                if entity.closed:
                    layer = entity.dxf.layer
                    calques.add(layer)
                    self.zones.append({
                        'points': [(p[0], p[1]) for p in entity],
                        'layer': layer
                    })

            for name in sorted(calques):
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.calques_list.addItem(item)

            self.mettre_a_jour_calques()
            self.status_bar.setText(f"{len(self.zones)} zones chargées.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur DXF", f"Erreur de lecture:\n{str(e)}")

    def mettre_a_jour_calques(self):
        """
        Met à jour les calques visibles.
        """
        self.calques_visibles = {
            self.calques_list.item(i).text()
            for i in range(self.calques_list.count())
            if self.calques_list.item(i).checkState() == Qt.CheckState.Checked
        }
        self.dessiner()

    def placer_ilots(self):
        """
        Place les îlots dans les zones disponibles.
        """
        self.ilots.clear()
        l = self.longueur_ilot.value() / self.echelle_dxf.value()
        h = self.largeur_ilot.value() / self.echelle_dxf.value()
        marge = self.marge.value() / self.echelle_dxf.value()

        for zone in self.zones:
            if zone['layer'] not in self.calques_visibles:
                continue

            pts = zone['points']
            path = MplPath(pts)
            xs, ys = zip(*pts)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            x = min_x + marge
            while x + l <= max_x - marge:
                y = min_y + marge
                while y + h <= max_y - marge:
                    center = (x + l/2, y + h/2)
                    rect = [(x, y), (x + l, y), (x + l, y + h), (x, y + h)]
                    if path.contains_point(center):
                        self.ilots.append(rect)
                    y += h + marge
                x += l + marge

        self.mettre_a_jour_stats()
        self.dessiner()

    def mettre_a_jour_stats(self):
        """
        Met à jour les statistiques.
        """
        nb = len(self.ilots)
        surface = nb * self.longueur_ilot.value() * self.largeur_ilot.value()
        total = surface * self.prix_m2.value()
        self.stats = {
            "nb": nb,
            "surface": surface,
            "cout": total
        }
        self.stats_label.setText(
            f"<b>Îlots placés :</b> {nb}<br>"
            f"<b>Surface totale :</b> {surface:.2f} m²<br>"
            f"<b>Prix estimé :</b> {total:,.0f} €"
        )
        self.status_bar.setText(f"{nb} îlots | {surface:.2f} m² | {total:,.0f} €")

    def dessiner(self):
        """
        Affiche les zones et îlots.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_aspect("equal")

        for zone in self.zones:
            if zone['layer'] in self.calques_visibles:
                poly = Polygon(zone['points'], closed=True, edgecolor='blue', fill=False)
                ax.add_patch(poly)

        for rect in self.ilots:
            ax.add_patch(Rectangle(rect[0],
                                   rect[1][0] - rect[0][0],
                                   rect[2][1] - rect[1][1],
                                   facecolor='green', alpha=0.5))

        ax.autoscale_view()
        ax.invert_yaxis()
        self.canvas.draw()

    def afficher_3d(self):
        """
        Affiche une vue 3D des îlots.
        """
        if not self.ilots:
            QMessageBox.warning(self, "Aucun îlot", "Placez d'abord des îlots.")
            return

        fig = Figure()
        ax = fig.add_subplot(111, projection='3d')

        for rect in self.ilots:
            x, y = rect[0]
            lx = rect[1][0] - rect[0][0]
            ly = rect[2][1] - rect[1][1]
            zz = [0, 0, 1, 1]
            xx = [x, x+lx, x+lx, x]
            yy = [y, y, y+ly, y+ly]
            ax.plot_trisurf(xx, yy, zz, color='green', alpha=0.6)

        ax.set_title("Vue 3D des îlots")
        win = QMainWindow()
        win.setWindowTitle("Vue 3D")
        canvas3d = FigureCanvasQTAgg(fig)
        win.setCentralWidget(canvas3d)
        win.resize(800, 600)
        win.show()

    def export_pdf(self):
        """
        Exporte les résultats en PDF.
        """
        if not self.ilots:
            QMessageBox.warning(self, "Rien à exporter", "Aucun îlot placé.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter PDF", "", "PDF (*.pdf)")
        if path:
            with PdfPages(path) as pdf:
                self.figure.savefig(pdf)
                fig2 = Figure()
                ax2 = fig2.add_subplot(111)
                ax2.axis("off")
                text = f"Îlots: {self.stats['nb']}\nSurface: {self.stats['surface']:.2f} m²\nCoût: {self.stats['cout']:.0f} €"
                ax2.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
                pdf.savefig(fig2)
            QMessageBox.information(self, "Export PDF", f"PDF exporté : {path}")

    def sauvegarder_projet(self):
        """
        Sauvegarde le projet actuel.
        """
        path, _ = QFileDialog.getSaveFileName(self, "Sauvegarder projet", "", "Projet JSON (*.json)")
        if path:
            data = {
                "longueur_ilot": self.longueur_ilot.value(),
                "largeur_ilot": self.largeur_ilot.value(),
                "marge": self.marge.value(),
                "prix_m2": self.prix_m2.value(),
                "echelle_dxf": self.echelle_dxf.value()
            }
            with open(path, "w") as f:
                json.dump(data, f)
            QMessageBox.information(self, "Projet", "Projet sauvegardé")

    def charger_projet(self):
        """
        Charge un projet sauvegardé.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Charger projet", "", "Projet JSON (*.json)")
        if path:
            with open(path) as f:
                data = json.load(f)
            self.longueur_ilot.setValue(data.get("longueur_ilot", 2.0))
            self.largeur_ilot.setValue(data.get("largeur_ilot", 1.5))
            self.marge.setValue(data.get("marge", 0.5))
            self.prix_m2.setValue(data.get("prix_m2", 1200))
            self.echelle_dxf.setValue(data.get("echelle_dxf", 1.0))
            QMessageBox.information(self, "Projet", "Projet chargé.")

    def reinitialiser(self):
        """
        Réinitialise l'application.
        """
        self.zones.clear()
        self.ilots.clear()
        self.calques_list.clear()
        self.figure.clear()
        self.canvas.draw()
        self.stats_label.setText("Aucun calcul effectué.")
        self.status_bar.setText("Réinitialisé")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = IlotPlannerPro()
    viewer.show()
    sys.exit(app.exec())
