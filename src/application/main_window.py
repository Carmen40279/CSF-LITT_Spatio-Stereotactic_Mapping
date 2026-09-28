from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget, QComboBox
)
from pyvistaqt import QtInteractor

from src.anatomy.csf import CSFCompartment
from src.anatomy.lesion import AnatomicalTarget
from src.application.case_manager import CaseManager
from src.application.case_processor import process_imaging_case
from src.application.state import ApplicationState
from src.geometry.vectors import Point3D
from src.planning.evaluation import evaluate_litt_trajectory
from src.planning.candidate_generation import CandidateGenerationConfig
from src.surgery.litt import LITTParameters, LITTPlan
from src.trajectory.trajectory import Trajectory
from src.surgery.seeg import SEEGElectrode
import inspect
from src.trajectory.ablation import AblationGeometry
from src.risk.mask_csf_exposure import calculate_mask_csf_exposure
from src.visualization.brain_scene import anatomical_mask_to_mesh, thermal_field_to_grid, create_temperature_isosurface


class LiveGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values=[]
        self.setMinimumHeight(150)

    def add_value(self, x, y):
        self.values.append((float(x), float(y)))
        self.values=self.values[-80:]
        self.update()

    def clear(self):
        self.values=[]
        self.update()

    def paintEvent(self, event):
        p=QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w=max(1,self.width()-45); h=max(1,self.height()-30); left=35; top=10; bottom=top+h
        p.setPen(QPen(Qt.gray)); p.drawLine(left,bottom,left,top); p.drawLine(left,bottom,left+w,bottom)
        p.setPen(QPen(Qt.white)); p.drawText(4,top+10,"100%"); p.drawText(10,bottom,"0")
        if len(self.values)<2:return
        xs=[v[0] for v in self.values]; ys=[v[1] for v in self.values]
        xmin,xmax=min(xs),max(xs)
        if abs(xmax-xmin)<1e-9:xmax=xmin+1
        pts=[]
        for x,y in self.values:
            px=left+(x-xmin)/(xmax-xmin)*w
            py=bottom-max(0.0,min(1.0,y))*h
            pts.append((px,py))
        p.setPen(QPen(Qt.cyan,2))
        for a,b in zip(pts,pts[1:]):p.drawLine(int(a[0]),int(a[1]),int(b[0]),int(b[1]))
        p.setPen(QPen(Qt.white)); p.drawText(left,bottom+20,f"offset {xmin:.0f}…{xmax:.0f} mm")


class PieWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exposure = 0.0
        self.setMinimumHeight(150)

    def set_exposure(self, value):
        self.exposure = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = min(self.width(), self.height()) - 30
        rect = (self.width()-r)//2, 10, r, r
        p.setBrush(QBrush(Qt.darkGray))
        p.setPen(QPen(Qt.NoPen))
        p.drawEllipse(*rect)
        p.setBrush(QBrush(Qt.cyan))
        p.drawPie(*rect, 90*16, int(-360*self.exposure*16))
        p.setPen(QPen(Qt.white))
        p.drawText(0, r+20, self.width(), 25, Qt.AlignCenter, f"CSF exposure {self.exposure*100:.1f}%")


def _point(v):
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)


def _distance_between_segments(a0, a1, b0, b1):
    # Closest distance between two finite 3-D segments.
    u = a1-a0; v=b1-b0; w=a0-b0
    a=np.dot(u,u); b=np.dot(u,v); c=np.dot(v,v); d=np.dot(u,w); e=np.dot(v,w)
    D=a*c-b*b
    sN=sD=D; tN=tD=D
    if D < 1e-12:
        sN=0.0; sD=1.0; tN=e; tD=c
    else:
        sN=b*e-c*d; tN=a*e-b*d
        if sN < 0:
            sN=0; tN=e; tD=c
        elif sN > sD:
            sN=sD; tN=e+b; tD=c
    if tN < 0:
        tN=0
        if -d < 0: sN=0
        elif -d > a: sN=sD
        else: sN=-d; sD=a
    elif tN > tD:
        tN=tD
        if -d+b < 0: sN=0
        elif -d+b > a: sN=sD
        else: sN=-d+b; sD=a
    sc=0 if abs(sN)<1e-12 else sN/sD
    tc=0 if abs(tN)<1e-12 else tN/tD
    delta=w+sc*u-tc*v
    return float(np.linalg.norm(delta))


def _distance_point_to_segment(point, a, b):
    ab=b-a; denom=float(np.dot(ab,ab))
    if denom<=1e-12:return float(np.linalg.norm(point-a))
    t=max(0.0,min(1.0,float(np.dot(point-a,ab)/denom)))
    return float(np.linalg.norm(point-(a+t*ab)))


class ThermoSEEGMainWindow(QMainWindow):
    """Interactive ThermoSEEG research-planning workstation."""

    def __init__(self):
        super().__init__()
        self.state = ApplicationState()
        self.case_manager = CaseManager(self.state)
        self.last_processing_result = None
        self.last_evaluation = None
        self.brain_mesh = None
        self.csf_mesh = None
        self.target_point = np.zeros(3)
        self.litt_entry = np.array([25.0, 0.0, 0.0])
        self.litt_target = np.zeros(3)
        self.seeg_entry = np.array([0.0, -30.0, 0.0])
        self.seeg_target = np.zeros(3)
        self.test_point = np.zeros(3)
        self.test_radius_value = 5.0
        self.live_history = LiveGraphWidget()
        self.thermal_visible = False
        self.seeg_object = None
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(250)
        self._live_timer.timeout.connect(self._apply_live_inputs)
        self._updating_fields = False
        self.litt_object = None
        self._build_interface()
        self._build_menu()
        self._connect_live_inputs()

    def _build_interface(self):
        self.setWindowTitle("ThermoSEEG — CSF-Aware LITT Planning Workstation")
        self.resize(1650, 950)
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central)
        root.addWidget(self._build_case_panel(), 1)
        root.addWidget(self._build_viewer(), 4)
        root.addWidget(self._build_planning_panel(), 1)
        self.statusBar().showMessage("Create/open a case, then process MRI.")

    def _connect_live_inputs(self):
        """Debounce numeric edits so the plan updates while the user works."""
        fields = [
            self.tx, self.ty, self.tz,
            self.test_x, self.test_y, self.test_z, self.test_radius,
            self.lx, self.ly, self.lz,
            self.ltargetx, self.ltargety, self.ltargetz,
            self.active, self.radius,
            self.ex, self.ey, self.ez,
            self.etx, self.ety, self.etz,
            self.contacts, self.spacing,
        ]
        for field in fields:
            field.textChanged.connect(self._schedule_live_update)

    def _schedule_live_update(self, *_):
        if self._updating_fields:
            return
        self._live_timer.start()

    def _apply_live_inputs(self):
        """Apply valid edits without forcing the user to press an Apply button."""
        if self.state.current_case is None:
            return
        try:
            target = self._read_point(self.tx, self.ty, self.tz)
            litt_entry = self._read_point(self.lx, self.ly, self.lz)
            litt_target = self._read_point(self.ltargetx, self.ltargety, self.ltargetz)
            seeg_entry = self._read_point(self.ex, self.ey, self.ez)
            seeg_target = self._read_point(self.etx, self.ety, self.etz)
            active = float(self.active.text())
            radius = float(self.radius.text())
            contacts = int(float(self.contacts.text()))
            spacing = float(self.spacing.text())
            test_point = self._read_point(self.test_x, self.test_y, self.test_z)
            test_radius = float(self.test_radius.text())
        except (ValueError, TypeError):
            return
        if active <= 0 or radius <= 0 or contacts < 1 or spacing <= 0 or test_radius <= 0:
            return
        self.target_point = target
        self.litt_entry = litt_entry
        self.litt_target = litt_target
        self.seeg_entry = seeg_entry
        self.seeg_target = seeg_target
        self.test_point = test_point
        self.test_radius_value = test_radius
        self._redraw_objects()
        self._live_metrics()

    def _build_menu(self):
        menu=self.menuBar().addMenu("File")
        a=menu.addAction("New Case"); a.triggered.connect(self._new_case)
        a=menu.addAction("Open Case"); a.triggered.connect(self._open_case)
        a=menu.addAction("Save Case"); a.triggered.connect(self._save_case)

    def _build_case_panel(self):
        panel=QWidget(); lay=QVBoxLayout(panel)
        g=QGroupBox("Case"); gl=QVBoxLayout(g)
        self.case_label=QLabel("No case loaded"); self.case_label.setWordWrap(True)
        for text,fn in [("New Case",self._new_case),("Open Case",self._open_case),("Save Case",self._save_case),("PROCESS MRI",self._process_mri)]:
            b=QPushButton(text); b.clicked.connect(fn); gl.addWidget(b)
        gl.addWidget(self.case_label); lay.addWidget(g)
        g=QGroupBox("Patient anatomy"); gl=QVBoxLayout(g)
        self.anatomy_label=QLabel("MRI: not loaded\nBrain: —\nCSF: —"); self.anatomy_label.setWordWrap(True); gl.addWidget(self.anatomy_label); lay.addWidget(g)
        g=QGroupBox("View"); gl=QVBoxLayout(g)
        for text,fn in [("3D",self._view_3d),("Axial",self._view_axial),("Coronal",self._view_coronal),("Sagittal",self._view_sagittal)]:
            b=QPushButton(text); b.clicked.connect(fn); gl.addWidget(b)
        lay.addWidget(g); lay.addStretch(); return panel

    def _build_viewer(self):
        panel=QWidget(); lay=QVBoxLayout(panel)
        self.plotter=QtInteractor(panel); self.plotter.set_background("#0b0f14"); lay.addWidget(self.plotter.interactor,1)
        row=QHBoxLayout();
        for text,fn in [("Reset camera",self._reset_view),("Pick target on brain",self._enable_target_picking),("Run thermal model",self._run_thermal)]:
            b=QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row); return panel

    def _build_planning_panel(self):
        outer=QScrollArea(); outer.setWidgetResizable(True); panel=QWidget(); lay=QVBoxLayout(panel)
        g=QGroupBox("Target structure"); f=QFormLayout(g)
        self.target_name=QLineEdit("Target"); self.tx=QLineEdit("0"); self.ty=QLineEdit("0"); self.tz=QLineEdit("0")
        self.test_name=QLineEdit("Test structure"); self.test_x=QLineEdit("10"); self.test_y=QLineEdit("0"); self.test_z=QLineEdit("0"); self.test_radius=QLineEdit("5")
        for lab,w in [("Name",self.test_name),("X mm",self.test_x),("Y mm",self.test_y),("Z mm",self.test_z),("Radius mm",self.test_radius)]: f.addRow(lab,w)
        b=QPushButton("Set test structure"); b.clicked.connect(self._apply_test_structure); f.addRow(b)
        for lab,w in [("Name",self.target_name),("X mm",self.tx),("Y mm",self.ty),("Z mm",self.tz)]: f.addRow(lab,w)
        b=QPushButton("Set target"); b.clicked.connect(self._apply_target); f.addRow(b); lay.addWidget(g)

        g=QGroupBox("LITT laser path"); f=QFormLayout(g)
        self.lx=QLineEdit("25"); self.ly=QLineEdit("0"); self.lz=QLineEdit("0")
        self.ltargetx=QLineEdit("0"); self.ltargety=QLineEdit("0"); self.ltargetz=QLineEdit("0")
        self.active=QLineEdit("25"); self.radius=QLineEdit("4")
        for lab,w in [("Entry X",self.lx),("Entry Y",self.ly),("Entry Z",self.lz),("Target X",self.ltargetx),("Target Y",self.ltargety),("Target Z",self.ltargetz),("Active length mm",self.active),("Radius mm",self.radius)]: f.addRow(lab,w)
        b=QPushButton("Insert / update LITT"); b.clicked.connect(self._apply_litt); f.addRow(b); lay.addWidget(g)

        g=QGroupBox("SEEG electrode"); f=QFormLayout(g)
        self.ex=QLineEdit("0"); self.ey=QLineEdit("-30"); self.ez=QLineEdit("0")
        self.etx=QLineEdit("0"); self.ety=QLineEdit("0"); self.etz=QLineEdit("0")
        self.contacts=QLineEdit("8"); self.spacing=QLineEdit("3.5")
        for lab,w in [("Entry X",self.ex),("Entry Y",self.ey),("Entry Z",self.ez),("Target X",self.etx),("Target Y",self.ety),("Target Z",self.etz),("Contacts",self.contacts),("Spacing mm",self.spacing)]: f.addRow(lab,w)
        b=QPushButton("Insert / update SEEG"); b.clicked.connect(self._apply_seeg); f.addRow(b); lay.addWidget(g)

        g=QGroupBox("Live trajectory controls"); v=QVBoxLayout(g)
        self.offset=QSlider(Qt.Horizontal); self.offset.setRange(-100,100); self.offset.setValue(0); self.offset.valueChanged.connect(self._live_offset)
        v.addWidget(QLabel("LITT lateral offset (mm)")); v.addWidget(self.offset)
        self.offset_value=QLabel("0.0 mm"); v.addWidget(self.offset_value)
        lay.addWidget(g)

        g=QGroupBox("ThermoSEEG analysis"); v=QVBoxLayout(g)
        self.pie=PieWidget(); v.addWidget(self.pie)
        self.metrics=QLabel("No analysis yet."); self.metrics.setWordWrap(True); v.addWidget(self.metrics)
        self.live_graph=QLabel("Live metric: —"); v.addWidget(self.live_graph)
        v.addWidget(QLabel("CSF exposure vs LITT offset"))
        v.addWidget(self.live_history)
        lay.addWidget(g); lay.addStretch()
        outer.setWidget(panel); return outer

    def _new_case(self):
        case = self.case_manager.new_case("THERMOSEEG-CASE-001")
        self.state.set_case(case)
        self.state.mark_dirty()
        self._clear_scene()
        self._refresh_labels()
        self.statusBar().showMessage("New ThermoSEEG case created.")

    def _refresh_labels(self):
        """Refresh case and anatomy summaries after case changes."""
        case = self.state.current_case
        if case is None:
            self.case_label.setText("No case loaded")
            self.anatomy_label.setText("MRI: not loaded\nBrain: —\nCSF: —")
            return
        result = self.last_processing_result
        if result is None:
            self.case_label.setText(case.case_id)
            self.anatomy_label.setText("MRI: not processed\nBrain: —\nCSF: —")
            return
        self.case_label.setText(f"{case.case_id}\n{Path(result.source_path).name}")
        self.anatomy_label.setText(
            f"MRI: VALID\nShape: {result.image_shape}\n"
            f"Spacing: {result.voxel_spacing}\n"
            f"Brain: {result.brain_volume_mm3:,.0f} mm³\n"
            f"CSF: {result.csf_volume_mm3:,.0f} mm³"
        )

    def _open_case(self):
        path,_=QFileDialog.getOpenFileName(self,"Open ThermoSEEG Case","","ThermoSEEG Case (*.json)")
        if not path:return
        try:
            case=self.case_manager.open_case(path); self.state.set_case(case); self.state.case_path=path; self.state.dirty=False; self._refresh_labels(); self.statusBar().showMessage(f"Opened {Path(path).name}")
        except Exception as exc: QMessageBox.critical(self,"Open Case Failed",str(exc))

    def _save_case(self):
        if self.state.current_case is None: return
        path=self.state.case_path
        if not path: path,_=QFileDialog.getSaveFileName(self,"Save ThermoSEEG Case","","ThermoSEEG Case (*.json)")
        if not path:return
        try:
            self.case_manager.save_case(self.state.current_case,path); self.state.case_path=path; self.state.dirty=False; self.statusBar().showMessage(f"Saved {Path(path).name}")
        except Exception as exc: QMessageBox.critical(self,"Save Case Failed",str(exc))

    def _process_mri(self):
        if self.state.current_case is None: self._new_case()
        path,_=QFileDialog.getOpenFileName(self,"Select MRI","data/sample","NIfTI MRI (*.nii *.nii.gz)")
        if not path:return
        try:
            self.statusBar().showMessage("Processing MRI anatomy...")
            QApplication.processEvents()
            # MRI import is for the interactive workstation. Keep automatic candidate
            # analysis tiny here; full thermal analysis is launched explicitly.
            quick_config = CandidateGenerationConfig(
                entry_spacing=3.0,
                search_radius=9.0,
                active_length=25.0,
                radius=4.0,
                max_candidates=1,
            )
            result = process_imaging_case(
                self.state.current_case,
                path,
                candidate_config=quick_config,
            )
            self.last_processing_result = result
            self._display_case(result)
            self.statusBar().showMessage("MRI loaded into ThermoSEEG workspace.")
        except Exception as exc: QMessageBox.critical(self,"MRI Processing Failed",str(exc))

    def _display_case(self,result):
        case=self.state.current_case
        self.case_label.setText(f"{case.case_id}\n{Path(result.source_path).name}")
        self.anatomy_label.setText(f"MRI: VALID\nShape: {result.image_shape}\nSpacing: {result.voxel_spacing}\nBrain: {result.brain_volume_mm3:,.0f} mm³\nCSF: {result.csf_volume_mm3:,.0f} mm³")
        self.target_point=np.array([result.target.x,result.target.y,result.target.z],float); self.litt_target=self.target_point.copy(); self.seeg_target=self.target_point.copy(); self.test_point=self.target_point.copy()
        self._set_fields(self.target_point,self.tx,self.ty,self.tz); self._set_fields(self.litt_target,self.ltargetx,self.ltargety,self.ltargetz); self._set_fields(self.seeg_target,self.etx,self.ety,self.etz); self._set_fields(self.test_point,self.test_x,self.test_y,self.test_z)
        self.litt_entry=self.target_point+np.array([25,0,0],float); self._set_fields(self.litt_entry,self.lx,self.ly,self.lz)
        self.seeg_entry=self.target_point+np.array([0,-30,0],float); self._set_fields(self.seeg_entry,self.ex,self.ey,self.ez)
        self._show_anatomy(); self._apply_target(); self._apply_litt(); self._apply_seeg()

    def _show_anatomy(self):
        """Build anatomy meshes once, cache them, then reuse them for interaction."""
        case = self.state.current_case
        self.plotter.clear()
        if case is None:
            return
        if self.brain_mesh is None and case.anatomy.has_surface("brain"):
            mesh = anatomical_mask_to_mesh(case.anatomy.get_mask("brain"))
            if mesh.n_faces > 120_000:
                try:
                    reduction = max(0.0, min(0.92, 1.0 - 120_000 / float(mesh.n_faces)))
                    mesh = mesh.decimate_pro(reduction, preserve_topology=True)
                except Exception:
                    pass
            self.brain_mesh = mesh
        if self.csf_mesh is None and case.anatomy.has_mask("csf"):
            mesh = anatomical_mask_to_mesh(case.anatomy.get_mask("csf"))
            if mesh.n_faces > 60_000:
                try:
                    reduction = max(0.0, min(0.92, 1.0 - 60_000 / float(mesh.n_faces)))
                    mesh = mesh.decimate_pro(reduction, preserve_topology=True)
                except Exception:
                    pass
            self.csf_mesh = mesh
        if self.brain_mesh is not None:
            self.plotter.add_mesh(self.brain_mesh, color="lightgray", opacity=0.18, smooth_shading=True, name="brain")
        if self.csf_mesh is not None:
            self.plotter.add_mesh(self.csf_mesh, color="deepskyblue", opacity=0.12, name="csf")
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()
        self._redraw_objects()

    def _clear_scene(self):
        self.plotter.clear()
        self.last_evaluation = None
        self.last_processing_result = None
        self.brain_mesh = None
        self.csf_mesh = None
        self.thermal_visible = False
        self.live_history.clear()
        self.metrics.setText("No analysis yet.")
        self.live_graph.setText("Live metric: —")
        self.pie.set_exposure(0)
        self.plotter.render()

    def _set_fields(self,p,a,b,c): a.setText(f"{p[0]:.3f}"); b.setText(f"{p[1]:.3f}"); c.setText(f"{p[2]:.3f}")
    def _read_point(self,a,b,c): return np.array([float(a.text()),float(b.text()),float(c.text())],float)

    def _apply_target(self):
        try:self.target_point=self._read_point(self.tx,self.ty,self.tz)
        except ValueError:return
        self._redraw_objects(); self._live_metrics()

    def _apply_test_structure(self):
        try:
            self.test_point=self._read_point(self.test_x,self.test_y,self.test_z)
            self.test_radius_value=float(self.test_radius.text())
        except ValueError as exc:
            QMessageBox.warning(self,"Invalid test structure",str(exc)); return
        self._redraw_objects(); self._live_metrics()

    def _apply_litt(self):
        try:
            self.litt_entry=self._read_point(self.lx,self.ly,self.lz); self.litt_target=self._read_point(self.ltargetx,self.ltargety,self.ltargetz)
        except ValueError as exc: QMessageBox.warning(self,"Invalid LITT input",str(exc)); return
        self._redraw_objects(); self._live_metrics()

    def _apply_seeg(self):
        try:self.seeg_entry=self._read_point(self.ex,self.ey,self.ez); self.seeg_target=self._read_point(self.etx,self.ety,self.etz)
        except ValueError:return
        try:
            traj=Trajectory(Point3D(*self.seeg_entry),Point3D(*self.seeg_target))
            sig=inspect.signature(SEEGElectrode)
            kwargs={}
            for name in sig.parameters:
                if name == "electrode_id": kwargs[name]="GUI-SEEG-001"
                elif name in ("trajectory",): kwargs[name]=traj
                elif name == "contact_count": kwargs[name]=int(self.contacts.text())
                elif name == "contact_spacing": kwargs[name]=float(self.spacing.text())
                elif name == "coordinate_space": kwargs[name]="patient"
            self.seeg_object=SEEGElectrode(**kwargs)
            try: self.state.current_case.add_seeg_plan(self.seeg_object)
            except Exception: pass
        except Exception:
            self.seeg_object=None
        self._redraw_objects(); self._live_metrics()

    def _live_offset(self,value):
        off=float(value); self.offset_value.setText(f"{off:.1f} mm")
        self.litt_entry=self.litt_entry.copy(); self.litt_entry[1]=self.target_point[1]+off
        self._set_fields(self.litt_entry,self.lx,self.ly,self.lz); self._redraw_objects(); self._live_metrics()

    def _redraw_objects(self, keep_thermal=False):
        if self.plotter is None:return
        object_names=["target","litt","ablation","seeg","contacts","test_structure"]
        if not keep_thermal:
            object_names += ["thermal","iso40","iso45","iso50","iso55"]
        for name in object_names:
            try:self.plotter.remove_actor(name)
            except Exception:pass
        # target
        self.plotter.add_mesh(pv.Sphere(radius=1.8,center=self.target_point),color="yellow",name="target")
        # Test / critical structure
        self.plotter.add_mesh(pv.Sphere(radius=self.test_radius_value,center=self.test_point),color="orange",opacity=.18,name="test_structure")
        # LITT
        lt=pv.Line(self.litt_entry,self.litt_target); self.plotter.add_mesh(lt,color="magenta",line_width=6,name="litt")
        try:
            ab=AblationGeometry(Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)),float(self.active.text()),float(self.radius.text()))
            cyl=pv.Cylinder(center=np.array(self.litt_target),direction=self.litt_entry-self.litt_target,radius=float(self.radius.text()),height=min(float(self.active.text()),np.linalg.norm(self.litt_entry-self.litt_target)))
            self.plotter.add_mesh(cyl,color="magenta",opacity=.25,name="ablation")
        except Exception:pass
        # SEEG
        se=pv.Line(self.seeg_entry,self.seeg_target); self.plotter.add_mesh(se,color="gold",line_width=5,name="seeg")
        pts=[]
        n=max(1,int(float(self.contacts.text()) if self.contacts.text() else 8)); spacing=float(self.spacing.text() or 3.5); d=self.seeg_target-self.seeg_entry; L=np.linalg.norm(d); u=d/L if L else np.array([0,0,1])
        for i in range(n): pts.append(self.seeg_target-u*(i*spacing))
        if pts:self.plotter.add_mesh(pv.PolyData(np.array(pts)),color="red",point_size=12,render_points_as_spheres=True,name="contacts")
        self.plotter.render()

    def _live_metrics(self):
        case=self.state.current_case
        if case is None or not case.anatomy.has_mask("csf"): return
        try:
            traj=Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)); ab=AblationGeometry(traj,float(self.active.text()),float(self.radius.text()))
            assessment=calculate_mask_csf_exposure(ab,case.anatomy.get_mask("csf")); exp=float(assessment.exposure_fraction)
            clearance=_distance_between_segments(self.litt_entry,self.litt_target,self.seeg_entry,self.seeg_target)
            target_dist=float(np.linalg.norm(self.litt_target-self.target_point))
            test_dist=max(0.0, _distance_point_to_segment(self.test_point,self.litt_entry,self.litt_target)-self.test_radius_value)
            self.pie.set_exposure(exp)
            self.live_history.add_value(float(self.offset.value()), exp)
            self.live_graph.setText(f"LIVE\nCSF exposure: {exp*100:.1f}%\nLITT→target distance: {target_dist:.2f} mm\nLITT↔SEEG clearance: {clearance:.2f} mm\nLITT→test-structure clearance: {test_dist:.2f} mm")
        except Exception as exc:self.live_graph.setText(f"Live metric unavailable: {exc}")

    def _run_thermal(self):
        case=self.state.current_case
        if case is None or not case.anatomy.has_mask("csf"):
            QMessageBox.warning(self,"No patient anatomy","Process an MRI first."); return
        try:
            target=AnatomicalTarget(self.target_name.text(),Point3D(*self.target_point),"patient","manual")
            litt=LITTPlan("GUI-LITT-001",Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)),LITTParameters(float(self.active.text()),float(self.radius.text())))
            compat=CSFCompartment(center=Point3D(*self.target_point),radii=(18,18,18),resolution=48)
            ev=evaluate_litt_trajectory(litt,compat,target=target,seeg_electrodes=[self.seeg_object] if self.seeg_object is not None else [],uncertainty_enabled=True,patient_csf_mask=case.anatomy.get_mask("csf"))
            self.litt_object=litt
            try: case.add_litt_plan(litt)
            except Exception: pass
            self.last_evaluation=ev; self._display_evaluation(ev); self._display_thermal(ev.thermal); self.statusBar().showMessage("ThermoSEEG thermal model completed.")
        except Exception as exc: QMessageBox.critical(self,"Thermal analysis failed",str(exc))

    def _display_evaluation(self,ev):
        exp=ev.csf_exposure_fraction; t=ev.thermal
        if t is None:return
        self.pie.set_exposure(exp)
        self.metrics.setText(f"TARGET\nDistance: {ev.target.target_distance:.2f} mm\n\nCSF\nExposure: {exp*100:.2f}%\nMinimum distance: {ev.minimum_csf_distance:.2f} mm\n\nTHERMAL\nPeak with CSF: {t.maximum_temperature:.2f} °C\nPeak without CSF: {t.maximum_temperature_without_csf:.2f} °C\nPeak reduction: {t.maximum_csf_temperature_reduction:.2f} °C\n\nDOSE\nWith CSF: {t.maximum_thermal_dose:.3f}\nWithout CSF: {t.maximum_thermal_dose_without_csf:.3f}\nDose reduction: {t.csf_thermal_dose_reduction:.3f}\n\nUNCERTAINTY\nTrajectories: {ev.uncertainty.number_of_trajectories if ev.uncertainty else 0}")

    def _display_thermal(self, thermal):
        if thermal is None:
            return
        for name in ["thermal", "iso40", "iso45", "iso50", "iso55"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        grid = thermal_field_to_grid(thermal, "temperature")
        peak = float(thermal.maximum_temperature)
        self.plotter.add_volume(
            grid,
            scalars="temperature",
            opacity=[0.0, 0.0, 0.03, 0.10, 0.20, 0.34],
            cmap="inferno",
            clim=(37.0, max(45.0, peak)),
            name="thermal",
        )
        try:
            self.plotter.add_scalar_bar(title="Modeled temperature (°C)", n_labels=5)
        except Exception:
            pass
        for temp, name in [(40.0, "iso40"), (45.0, "iso45"), (50.0, "iso50"), (55.0, "iso55")]:
            if peak < temp:
                continue
            try:
                iso = create_temperature_isosurface(thermal, temp)
                if iso is not None and iso.n_points:
                    self.plotter.add_mesh(iso, opacity=0.28, name=name, smooth_shading=True)
            except Exception:
                pass
        self.thermal_visible = True
        self._redraw_objects(keep_thermal=True)
        self.plotter.render()

    def _enable_target_picking(self):
        try:
            self.plotter.enable_surface_point_picking(callback=self._picked_target,show_message=True,show_point=True,clear_on_no_selection=False)
            self.statusBar().showMessage("Click the brain surface to place the target.")
        except Exception as exc: QMessageBox.warning(self,"Picking unavailable",str(exc))

    def _picked_target(self,point):
        p=np.asarray(point,dtype=float); self.target_point=p; self._set_fields(p,self.tx,self.ty,self.tz); self._set_fields(p,self.ltargetx,self.ltargety,self.ltargetz); self._set_fields(p,self.etx,self.ety,self.etz); self._redraw_objects(); self._live_metrics(); self.statusBar().showMessage(f"Target placed at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")

    def _reset_view(self): self.plotter.reset_camera(); self.plotter.render()
    def _view_3d(self): self.plotter.reset_camera(); self.plotter.render()
    def _view_axial(self): self.plotter.view_xy(); self.plotter.render()
    def _view_coronal(self): self.plotter.view_xz(); self.plotter.render()
    def _view_sagittal(self): self.plotter.view_yz(); self.plotter.render()

    def closeEvent(self,event):
        try:self.plotter.close()
        except Exception:pass
        event.accept()


def launch_application():
    app=QApplication.instance() or QApplication([]); window=ThermoSEEGMainWindow(); window.show(); app.exec()


if __name__ == "__main__": launch_application()
