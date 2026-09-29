from __future__ import annotations
from pathlib import Path
import inspect
import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget, QComboBox, QCheckBox
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
from src.trajectory.ablation import AblationGeometry
from src.risk.mask_csf_exposure import calculate_mask_csf_exposure
from src.risk.thermal_field import calculate_thermal_field
from src.visualization.brain_scene import (
    anatomical_mask_to_mesh, thermal_field_to_grid,
    create_temperature_isosurface,
)

class LiveGraphWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.values=[]; self.setMinimumHeight(150)
    def add_value(self,x,y):
        self.values.append((float(x),float(y))); self.values=self.values[-80:]; self.update()
    def clear(self): self.values=[]; self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w=max(1,self.width()-45); h=max(1,self.height()-30); left=35; top=10; bottom=top+h
        p.setPen(QPen(Qt.gray)); p.drawLine(left,bottom,left,top); p.drawLine(left,bottom,left+w,bottom)
        p.setPen(QPen(Qt.white)); p.drawText(4,top+10,"100%"); p.drawText(10,bottom,"0")
        if len(self.values)<2:return
        xs=[x for x,y in self.values]; ys=[y for x,y in self.values]; xmin,xmax=min(xs),max(xs)
        if abs(xmax-xmin)<1e-9:xmax=xmin+1
        pts=[(left+(x-xmin)/(xmax-xmin)*w,bottom-max(0,min(1,y))*h) for x,y in self.values]
        p.setPen(QPen(Qt.cyan,2))
        for a,b in zip(pts,pts[1:]):p.drawLine(int(a[0]),int(a[1]),int(b[0]),int(b[1]))
        p.setPen(QPen(Qt.white)); p.drawText(left,bottom+20,f"offset {xmin:.0f}…{xmax:.0f} mm")

class PieWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.exposure=0.; self.setMinimumHeight(150)
    def set_exposure(self,v): self.exposure=max(0,min(1,float(v))); self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); r=min(self.width(),self.height())-30
        rect=(self.width()-r)//2,10,r,r; p.setBrush(QBrush(Qt.darkGray)); p.setPen(QPen(Qt.NoPen)); p.drawEllipse(*rect)
        p.setBrush(QBrush(Qt.cyan)); p.drawPie(*rect,90*16,int(-360*self.exposure*16))
        p.setPen(QPen(Qt.white)); p.drawText(0,r+20,self.width(),25,Qt.AlignCenter,f"CSF exposure {self.exposure*100:.1f}%")

def _distance_between_segments(a0,a1,b0,b1):
    u=a1-a0; v=b1-b0; w=a0-b0; a=np.dot(u,u); b=np.dot(u,v); c=np.dot(v,v); d=np.dot(u,w); e=np.dot(v,w); D=a*c-b*b
    sN=sD=D; tN=tD=D
    if D<1e-12:sN=0.; sD=1.; tN=e; tD=c
    else:
        sN=b*e-c*d; tN=a*e-b*d
        if sN<0:sN=0.; tN=e; tD=c
        elif sN>sD:sN=sD; tN=e+b; tD=c
    if tN<0:
        tN=0.
        if -d<0:sN=0.
        elif -d>a:sN=sD
        else:sN=-d; sD=a
    elif tN>tD:
        tN=tD
        if -d+b<0:sN=0.
        elif -d+b>a:sN=sD
        else:sN=-d+b; sD=a
    sc=0 if abs(sN)<1e-12 else sN/sD; tc=0 if abs(tN)<1e-12 else tN/tD
    return float(np.linalg.norm(w+sc*u-tc*v))

def _distance_point_to_segment(point,a,b):
    ab=b-a; den=float(np.dot(ab,ab))
    if den<=1e-12:return float(np.linalg.norm(point-a))
    t=max(0,min(1,float(np.dot(point-a,ab)/den))); return float(np.linalg.norm(point-(a+t*ab)))

class ThermoSEEGMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state=ApplicationState(); self.case_manager=CaseManager(self.state)
        self.last_processing_result=None; self.last_evaluation=None
        self.brain_mesh=None; self.csf_mesh=None
        self.target_point=np.zeros(3); self.litt_entry=np.array([25.,0.,0.]); self.litt_target=np.zeros(3)
        self.seeg_entry=np.array([0.,-30.,0.]); self.seeg_target=np.zeros(3)
        self.test_point=np.zeros(3); self.test_radius_value=5.
        self.seeg_object=None; self.litt_object=None; self.current_thermal=None; self.thermal_visible=False
        self.live_history=LiveGraphWidget(); self.candidate_records=[]; self._updating_fields=False
        self._live_timer=QTimer(self); self._live_timer.setSingleShot(True); self._live_timer.setInterval(160); self._live_timer.timeout.connect(self._apply_live_inputs)
        self._thermal_timer=QTimer(self); self._thermal_timer.setSingleShot(True); self._thermal_timer.setInterval(400); self._thermal_timer.timeout.connect(self._update_live_thermal)
        self._build_interface(); self._build_menu(); self._connect_live_inputs()

    def _build_interface(self):
        self.setWindowTitle("ThermoSEEG — CSF-Aware LITT Planning Workstation"); self.resize(1650,950)
        c=QWidget(); self.setCentralWidget(c); root=QHBoxLayout(c)
        root.addWidget(self._build_case_panel(),1); root.addWidget(self._build_viewer(),4); root.addWidget(self._build_planning_panel(),1)
        self.statusBar().showMessage("Create/open a case, process MRI, then move the LITT path.")

    def _connect_live_inputs(self):
        fields=[self.tx,self.ty,self.tz,self.test_x,self.test_y,self.test_z,self.test_radius,self.lx,self.ly,self.lz,
                self.ltargetx,self.ltargety,self.ltargetz,self.active,self.radius,self.ex,self.ey,self.ez,self.etx,
                self.ety,self.etz,self.contacts,self.spacing,self.laser_power,self.sim_time,self.csf_sink]
        for f in fields:f.textChanged.connect(self._schedule_live_update)
        self.thermal_mode.currentTextChanged.connect(lambda *_:self._schedule_thermal_preview())
        self.live_thermal.toggled.connect(lambda *_:self._schedule_thermal_preview())

    def _schedule_live_update(self,*_):
        if not self._updating_fields:self._live_timer.start()
    def _schedule_thermal_preview(self):
        if self.live_thermal.isChecked() and self.state.current_case is not None:self._thermal_timer.start()

    def _apply_live_inputs(self):
        if self.state.current_case is None:return
        try:
            target=self._read_point(self.tx,self.ty,self.tz); le=self._read_point(self.lx,self.ly,self.lz)
            lt=self._read_point(self.ltargetx,self.ltargety,self.ltargetz); se=self._read_point(self.ex,self.ey,self.ez)
            st=self._read_point(self.etx,self.ety,self.etz); active=float(self.active.text()); radius=float(self.radius.text())
            contacts=int(float(self.contacts.text())); spacing=float(self.spacing.text()); test=self._read_point(self.test_x,self.test_y,self.test_z)
            tr=float(self.test_radius.text())
        except (ValueError,TypeError):return
        if min(active,radius,spacing,tr)<=0 or contacts<1:return
        self.target_point=target; self.litt_entry=le; self.litt_target=lt; self.seeg_entry=se; self.seeg_target=st
        self.test_point=test; self.test_radius_value=tr
        self._apply_seeg_object_silent(); self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview()

    def _build_menu(self):
        m=self.menuBar().addMenu("File")
        for text,fn in [("New Case",self._new_case),("Open Case",self._open_case),("Save Case",self._save_case)]:
            a=m.addAction(text); a.triggered.connect(fn)

    def _build_case_panel(self):
        p=QWidget(); l=QVBoxLayout(p); g=QGroupBox("Case"); gl=QVBoxLayout(g); self.case_label=QLabel("No case loaded"); self.case_label.setWordWrap(True)
        for text,fn in [("New Case",self._new_case),("Open Case",self._open_case),("Save Case",self._save_case),("PROCESS MRI",self._process_mri)]:
            b=QPushButton(text); b.clicked.connect(fn); gl.addWidget(b)
        gl.addWidget(self.case_label); l.addWidget(g)
        g=QGroupBox("Patient anatomy"); gl=QVBoxLayout(g); self.anatomy_label=QLabel("MRI: not loaded\nBrain: —\nCSF: —"); self.anatomy_label.setWordWrap(True); gl.addWidget(self.anatomy_label); l.addWidget(g)
        g=QGroupBox("View"); gl=QVBoxLayout(g)
        for text,fn in [("3D",self._view_3d),("Axial",self._view_axial),("Coronal",self._view_coronal),("Sagittal",self._view_sagittal)]:
            b=QPushButton(text); b.clicked.connect(fn); gl.addWidget(b)
        l.addWidget(g); l.addStretch(); return p

    def _build_viewer(self):
        p=QWidget(); l=QVBoxLayout(p); self.plotter=QtInteractor(p); self.plotter.set_background("#0b0f14"); l.addWidget(self.plotter.interactor,1)
        row=QHBoxLayout()
        for text,fn in [("Reset camera",self._reset_view),("Pick target on brain",self._enable_target_picking),("Run ThermoSEEG",self._run_thermal),("Capture candidate",self._capture_candidate)]:
            b=QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        l.addLayout(row); return p

    def _build_planning_panel(self):
        outer=QScrollArea(); outer.setWidgetResizable(True); panel=QWidget(); l=QVBoxLayout(panel)
        g=QGroupBox("Target & test structure"); f=QFormLayout(g)
        self.target_name=QLineEdit("Target"); self.tx=QLineEdit("0"); self.ty=QLineEdit("0"); self.tz=QLineEdit("0")
        self.test_name=QLineEdit("Test structure"); self.test_x=QLineEdit("10"); self.test_y=QLineEdit("0"); self.test_z=QLineEdit("0"); self.test_radius=QLineEdit("5")
        for lab,w in [("Target name",self.target_name),("Target X mm",self.tx),("Target Y mm",self.ty),("Target Z mm",self.tz),("Test name",self.test_name),("Test X mm",self.test_x),("Test Y mm",self.test_y),("Test Z mm",self.test_z),("Test radius mm",self.test_radius)]:f.addRow(lab,w)
        b=QPushButton("Set structures"); b.clicked.connect(self._apply_target); b.clicked.connect(self._apply_test_structure); f.addRow(b); l.addWidget(g)

        g=QGroupBox("LITT laser path"); f=QFormLayout(g)
        self.lx=QLineEdit("25"); self.ly=QLineEdit("0"); self.lz=QLineEdit("0"); self.ltargetx=QLineEdit("0"); self.ltargety=QLineEdit("0"); self.ltargetz=QLineEdit("0"); self.active=QLineEdit("25"); self.radius=QLineEdit("4")
        for lab,w in [("Entry X",self.lx),("Entry Y",self.ly),("Entry Z",self.lz),("Target X",self.ltargetx),("Target Y",self.ltargety),("Target Z",self.ltargetz),("Active length mm",self.active),("Radius mm",self.radius)]:f.addRow(lab,w)
        b=QPushButton("Insert / update LITT"); b.clicked.connect(self._apply_litt); f.addRow(b); l.addWidget(g)

        g=QGroupBox("SEEG electrode"); f=QFormLayout(g)
        self.ex=QLineEdit("0"); self.ey=QLineEdit("-30"); self.ez=QLineEdit("0"); self.etx=QLineEdit("0"); self.ety=QLineEdit("0"); self.etz=QLineEdit("0"); self.contacts=QLineEdit("8"); self.spacing=QLineEdit("3.5")
        for lab,w in [("Entry X",self.ex),("Entry Y",self.ey),("Entry Z",self.ez),("Target X",self.etx),("Target Y",self.ety),("Target Z",self.etz),("Contacts",self.contacts),("Spacing mm",self.spacing)]:f.addRow(lab,w)
        b=QPushButton("Insert / update SEEG"); b.clicked.connect(self._apply_seeg); f.addRow(b); l.addWidget(g)

        g=QGroupBox("Live trajectory controls"); v=QVBoxLayout(g); self.offset=QSlider(Qt.Horizontal); self.offset.setRange(-100,100); self.offset.setValue(0); self.offset.valueChanged.connect(self._live_offset); v.addWidget(QLabel("LITT lateral offset (mm)")); v.addWidget(self.offset); self.offset_value=QLabel("0.0 mm"); v.addWidget(self.offset_value)
        self.live_thermal=QCheckBox("Live thermal preview"); self.live_thermal.setChecked(True); v.addWidget(self.live_thermal); l.addWidget(g)

        g=QGroupBox("Thermal controls"); f=QFormLayout(g); self.laser_power=QLineEdit("10.0"); self.sim_time=QLineEdit("60.0"); self.csf_sink=QLineEdit("15000"); self.thermal_mode=QComboBox(); self.thermal_mode.addItems(["Temperature rise (ΔT)","Temperature","CSF heat-sink effect","Thermal dose"])
        for lab,w in [("Laser power W",self.laser_power),("Simulation time s",self.sim_time),("CSF sink strength",self.csf_sink),("Display",self.thermal_mode)]:f.addRow(lab,w)
        b=QPushButton("Update thermal field now"); b.clicked.connect(self._update_live_thermal); f.addRow(b); l.addWidget(g)

        g=QGroupBox("ThermoSEEG analysis"); v=QVBoxLayout(g); self.pie=PieWidget(); v.addWidget(self.pie)
        self.metrics=QLabel("No analysis yet."); self.metrics.setWordWrap(True); v.addWidget(self.metrics)
        self.live_graph=QLabel("Live metric: —"); self.live_graph.setWordWrap(True); v.addWidget(self.live_graph)
        v.addWidget(QLabel("CSF exposure vs LITT offset")); v.addWidget(self.live_history)
        self.candidate_label=QLabel("No captured candidates."); self.candidate_label.setWordWrap(True); v.addWidget(self.candidate_label)
        l.addWidget(g); l.addStretch(); outer.setWidget(panel); return outer

    def _new_case(self):
        case=self.case_manager.new_case("THERMOSEEG-CASE-001"); self.state.set_case(case); self.state.mark_dirty()
        self.last_processing_result=None; self.last_evaluation=None; self.candidate_records=[]; self._clear_scene(); self._refresh_labels()
        self.statusBar().showMessage("New ThermoSEEG case created.")

    def _refresh_labels(self):
        case=self.state.current_case
        if case is None:self.case_label.setText("No case loaded"); self.anatomy_label.setText("MRI: not loaded\nBrain: —\nCSF: —"); return
        result=self.last_processing_result
        if result is None:self.case_label.setText(case.case_id); self.anatomy_label.setText("MRI: not processed\nBrain: —\nCSF: —"); return
        self.case_label.setText(f"{case.case_id}\n{Path(result.source_path).name}")
        self.anatomy_label.setText(f"MRI: VALID\nShape: {result.image_shape}\nSpacing: {result.voxel_spacing}\nBrain: {result.brain_volume_mm3:,.0f} mm³\nCSF: {result.csf_volume_mm3:,.0f} mm³")

    def _open_case(self):
        path,_=QFileDialog.getOpenFileName(self,"Open ThermoSEEG Case","","ThermoSEEG Case (*.json)")
        if not path:return
        try:
            case=self.case_manager.open_case(path); self.state.set_case(case); self.state.case_path=path; self.state.dirty=False; self._refresh_labels(); self.statusBar().showMessage(f"Opened {Path(path).name}")
        except Exception as exc:QMessageBox.critical(self,"Open Case Failed",str(exc))

    def _save_case(self):
        if self.state.current_case is None:return
        path=self.state.case_path
        if not path:path,_=QFileDialog.getSaveFileName(self,"Save ThermoSEEG Case","","ThermoSEEG Case (*.json)")
        if not path:return
        try:
            self.case_manager.save_case(self.state.current_case,path); self.state.case_path=path; self.state.dirty=False; self.statusBar().showMessage(f"Saved {Path(path).name}")
        except Exception as exc:QMessageBox.critical(self,"Save Case Failed",str(exc))

    def _process_mri(self):
        if self.state.current_case is None:self._new_case()
        path,_=QFileDialog.getOpenFileName(self,"Select MRI","data/sample","NIfTI MRI (*.nii *.nii.gz)")
        if not path:return
        try:
            self.statusBar().showMessage("Processing MRI anatomy..."); QApplication.processEvents()
            cfg=CandidateGenerationConfig(entry_spacing=3.,search_radius=9.,active_length=25.,radius=4.,max_candidates=1)
            result=process_imaging_case(self.state.current_case,path,candidate_config=cfg); self.last_processing_result=result; self.candidate_records=[]; self._display_case(result)
            self.statusBar().showMessage("MRI loaded into ThermoSEEG workspace.")
        except Exception as exc:QMessageBox.critical(self,"MRI Processing Failed",str(exc))

    def _display_case(self,result):
        case=self.state.current_case; self.case_label.setText(f"{case.case_id}\n{Path(result.source_path).name}")
        self.anatomy_label.setText(f"MRI: VALID\nShape: {result.image_shape}\nSpacing: {result.voxel_spacing}\nBrain: {result.brain_volume_mm3:,.0f} mm³\nCSF: {result.csf_volume_mm3:,.0f} mm³")
        self.target_point=np.array([result.target.x,result.target.y,result.target.z],float); self.litt_target=self.target_point.copy(); self.seeg_target=self.target_point.copy(); self.test_point=self.target_point.copy()
        self._set_fields(self.target_point,self.tx,self.ty,self.tz); self._set_fields(self.litt_target,self.ltargetx,self.ltargety,self.ltargetz); self._set_fields(self.seeg_target,self.etx,self.ety,self.etz); self._set_fields(self.test_point,self.test_x,self.test_y,self.test_z)
        self.litt_entry=self.target_point+np.array([25.,0.,0.]); self._set_fields(self.litt_entry,self.lx,self.ly,self.lz); self.seeg_entry=self.target_point+np.array([0.,-30.,0.]); self._set_fields(self.seeg_entry,self.ex,self.ey,self.ez)
        self._show_anatomy(); self._apply_target(); self._apply_litt(); self._apply_seeg()

    def _show_anatomy(self):
        case=self.state.current_case; self.plotter.clear()
        if case is None:return
        if self.brain_mesh is None and case.anatomy.has_surface("brain"):
            mesh=anatomical_mask_to_mesh(case.anatomy.get_mask("brain"))
            if mesh.n_faces>120000:
                try:mesh=mesh.decimate_pro(max(0,min(.92,1-120000/float(mesh.n_faces))),preserve_topology=True)
                except Exception:pass
            self.brain_mesh=mesh
        if self.csf_mesh is None and case.anatomy.has_mask("csf"):
            mesh=anatomical_mask_to_mesh(case.anatomy.get_mask("csf"))
            if mesh.n_faces>60000:
                try:mesh=mesh.decimate_pro(max(0,min(.92,1-60000/float(mesh.n_faces))),preserve_topology=True)
                except Exception:pass
            self.csf_mesh=mesh
        if self.brain_mesh is not None:self.plotter.add_mesh(self.brain_mesh,color="lightgray",opacity=.18,smooth_shading=True,name="brain")
        if self.csf_mesh is not None:self.plotter.add_mesh(self.csf_mesh,color="deepskyblue",opacity=.12,name="csf")
        self.plotter.add_axes(); self.plotter.show_grid(); self.plotter.reset_camera(); self._redraw_objects()

    def _clear_scene(self):
        self._live_timer.stop(); self._thermal_timer.stop(); self.plotter.clear(); self.last_evaluation=None; self.last_processing_result=None; self.brain_mesh=None; self.csf_mesh=None; self.current_thermal=None; self.thermal_visible=False
        self.live_history.clear(); self.metrics.setText("No analysis yet."); self.live_graph.setText("Live metric: —"); self.pie.set_exposure(0); self.candidate_records=[]; self.candidate_label.setText("No captured candidates."); self.plotter.render()

    def _set_fields(self,p,a,b,c):
        self._updating_fields=True
        try:a.setText(f"{p[0]:.3f}"); b.setText(f"{p[1]:.3f}"); c.setText(f"{p[2]:.3f}")
        finally:self._updating_fields=False
    def _read_point(self,a,b,c):return np.array([float(a.text()),float(b.text()),float(c.text())],float)

    def _apply_target(self):
        try:self.target_point=self._read_point(self.tx,self.ty,self.tz)
        except ValueError:return
        self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview()

    def _apply_test_structure(self):
        try:self.test_point=self._read_point(self.test_x,self.test_y,self.test_z); self.test_radius_value=float(self.test_radius.text())
        except ValueError:return
        self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview()

    def _apply_litt(self):
        try:self.litt_entry=self._read_point(self.lx,self.ly,self.lz); self.litt_target=self._read_point(self.ltargetx,self.ltargety,self.ltargetz)
        except ValueError:return
        self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview()

    def _apply_seeg_object_silent(self):
        try:
            traj=Trajectory(Point3D(*self.seeg_entry),Point3D(*self.seeg_target)); sig=inspect.signature(SEEGElectrode); kw={}
            for name in sig.parameters:
                if name=="electrode_id":kw[name]="GUI-SEEG-001"
                elif name=="trajectory":kw[name]=traj
                elif name=="contact_count":kw[name]=int(float(self.contacts.text()))
                elif name=="contact_spacing":kw[name]=float(self.spacing.text())
                elif name=="coordinate_space":kw[name]="patient"
            self.seeg_object=SEEGElectrode(**kw)
        except Exception:self.seeg_object=None

    def _apply_seeg(self):
        try:self.seeg_entry=self._read_point(self.ex,self.ey,self.ez); self.seeg_target=self._read_point(self.etx,self.ety,self.etz)
        except ValueError:return
        self._apply_seeg_object_silent()
        try:
            if self.seeg_object is not None:self.state.current_case.add_seeg_plan(self.seeg_object)
        except Exception:pass
        self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview()

    def _live_offset(self,value):
        off=float(value); self.offset_value.setText(f"{off:.1f} mm"); self.litt_entry=self.litt_entry.copy(); self.litt_entry[1]=self.target_point[1]+off
        self._set_fields(self.litt_entry,self.lx,self.ly,self.lz); self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview()

    def _remove_thermal(self):
        for name in ["thermal","iso40","iso45","iso50","iso55"]:
            try:self.plotter.remove_actor(name)
            except Exception:pass
        for title in ["Thermal field","Modeled temperature (°C)","Temperature rise ΔT (°C)","CSF temperature reduction (°C)","Thermal dose"]:
            try:self.plotter.remove_scalar_bar(title)
            except Exception:pass

    def _redraw_objects(self,keep_thermal=False):
        names=["target","target_ring","litt","litt_entry","litt_target","ablation","seeg","seeg_entry","contacts","test_structure"]
        if not keep_thermal:self._remove_thermal()
        for n in names:
            try:self.plotter.remove_actor(n)
            except Exception:pass
        self.plotter.add_mesh(pv.Sphere(radius=2.2,center=self.target_point),color="yellow",name="target")
        self.plotter.add_mesh(pv.Sphere(radius=3.2,center=self.target_point),color="yellow",opacity=.15,name="target_ring")
        self.plotter.add_mesh(pv.Sphere(radius=self.test_radius_value,center=self.test_point),color="orange",opacity=.18,name="test_structure")
        self.plotter.add_mesh(pv.Line(self.litt_entry,self.litt_target),color="magenta",line_width=10,render_lines_as_tubes=True,name="litt")
        self.plotter.add_mesh(pv.Sphere(radius=2.4,center=self.litt_entry),color="white",name="litt_entry")
        self.plotter.add_mesh(pv.Sphere(radius=2.,center=self.litt_target),color="magenta",name="litt_target")
        try:
            active=float(self.active.text()); radius=float(self.radius.text()); d=self.litt_entry-self.litt_target; L=float(np.linalg.norm(d)); h=min(active,L)
            if h>0:
                center=self.litt_target+.5*h*d/max(L,1e-9)
                self.plotter.add_mesh(pv.Cylinder(center=center,direction=d,radius=radius,height=h),color="magenta",opacity=.30,smooth_shading=True,name="ablation")
        except Exception:pass
        self.plotter.add_mesh(pv.Line(self.seeg_entry,self.seeg_target),color="gold",line_width=8,render_lines_as_tubes=True,name="seeg")
        self.plotter.add_mesh(pv.Sphere(radius=2.,center=self.seeg_entry),color="white",name="seeg_entry")
        try:
            n=max(1,int(float(self.contacts.text()))); spacing=float(self.spacing.text() or 3.5); d=self.seeg_target-self.seeg_entry; L=np.linalg.norm(d); u=d/L if L else np.array([0.,0.,1.])
            pts=[self.seeg_target-u*(i*spacing) for i in range(n)]
            self.plotter.add_mesh(pv.PolyData(np.asarray(pts)),color="red",point_size=15,render_points_as_spheres=True,name="contacts")
        except Exception:pass
        if keep_thermal:self._restore_current_thermal()
        self.plotter.render()

    def _live_metrics(self):
        case=self.state.current_case
        if case is None or not case.anatomy.has_mask("csf"):return
        try:
            ab=AblationGeometry(Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)),float(self.active.text()),float(self.radius.text()))
            exp=float(calculate_mask_csf_exposure(ab,case.anatomy.get_mask("csf")).exposure_fraction)
            clearance=_distance_between_segments(self.litt_entry,self.litt_target,self.seeg_entry,self.seeg_target)
            td=float(np.linalg.norm(self.litt_target-self.target_point)); test=max(0.,_distance_point_to_segment(self.test_point,self.litt_entry,self.litt_target)-self.test_radius_value)
            self.pie.set_exposure(exp); self.live_history.add_value(float(self.offset.value()),exp)
            extra=""
            if self.current_thermal is not None:
                t=self.current_thermal; extra=f"\n\nTHERMAL\nPeak ΔT: {t.maximum_temperature-getattr(t,'baseline_temperature_celsius',37.):.2f} °C\nCSF heat-sink reduction: {t.maximum_csf_temperature_reduction:.2f} °C"
            self.live_graph.setText(f"LIVE PLAN\nCSF exposure: {exp*100:.1f}%\nLITT → target: {td:.2f} mm\nLITT ↔ SEEG: {clearance:.2f} mm\nLITT → test structure: {test:.2f} mm{extra}")
        except Exception as exc:self.live_graph.setText(f"Live metric unavailable: {exc}")

    def _thermal_params(self):
        return float(self.laser_power.text()),float(self.sim_time.text()),float(self.csf_sink.text())

    def _update_live_thermal(self):
        if not self.live_thermal.isChecked():return
        case=self.state.current_case
        if case is None or not case.anatomy.has_mask("csf"):return
        try:
            power,sim,sink=self._thermal_params()
            if power<=0 or sim<=0 or sink<0:return
            self.statusBar().showMessage("Updating live thermal field..."); QApplication.processEvents()
            ab=AblationGeometry(Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)),float(self.active.text()),float(self.radius.text()))
            thermal=calculate_thermal_field(ab,CSFCompartment(center=Point3D(*self.target_point),radii=(18,18,18),resolution=24),
                patient_csf_mask=case.anatomy.get_mask("csf"),grid_resolution=20,margin=10.,laser_power_watts=power,
                simulation_time_seconds=min(sim,8.),time_step_seconds=1.,csf_sink_strength=sink)
            self.current_thermal=thermal; self.thermal_visible=True; self._display_thermal(thermal,True); self._live_metrics()
            self.statusBar().showMessage("Live thermal field updated.")
        except (ValueError,TypeError):return
        except Exception as exc:self.statusBar().showMessage(f"Thermal preview unavailable: {exc}")

    def _thermal_grid(self,t):
        mode=self.thermal_mode.currentText()
        if mode=="Temperature":return thermal_field_to_grid(t,"temperature"),"Modeled temperature (°C)","temperature"
        if mode=="CSF heat-sink effect":return thermal_field_to_grid(t,"csf_temperature_reduction"),"CSF temperature reduction (°C)","csf_temperature_reduction"
        if mode=="Thermal dose":return thermal_field_to_grid(t,"thermal_dose"),"Thermal dose","thermal_dose"
        g=thermal_field_to_grid(t,"temperature"); g["temperature_rise"]=np.asarray(g["temperature"])-getattr(t,"baseline_temperature_celsius",37.); return g,"Temperature rise ΔT (°C)","temperature_rise"

    def _display_thermal(self,thermal,live_preview=False):
        if thermal is None:return
        self._remove_thermal(); grid,title,key=self._thermal_grid(thermal); vals=np.asarray(grid[key],float); finite=vals[np.isfinite(vals)]
        if finite.size:
            low=float(np.percentile(finite,2)); high=float(np.percentile(finite,99))
            if high<=low:high=low+1.
            if key=="temperature":low=37.; high=max(45.,high)
            elif key in ("temperature_rise","csf_temperature_reduction","thermal_dose"):low=0.; high=max(1.,high)
            self.plotter.add_volume(grid,scalars=key,opacity=[0.,.015,.045,.10,.20,.34],cmap="inferno",clim=(low,high),name="thermal")
        try:self.plotter.add_scalar_bar(title=title,n_labels=5)
        except Exception:pass
        if self.thermal_mode.currentText() in ("Temperature","Temperature rise (ΔT)"):
            try:
                peak=float(thermal.maximum_temperature)
                for temp,name in [(40., "iso40"),(45.,"iso45"),(50.,"iso50"),(55.,"iso55")]:
                    if peak>=temp:
                        iso=create_temperature_isosurface(thermal,temp)
                        if iso is not None and iso.n_points:self.plotter.add_mesh(iso,opacity=.25,smooth_shading=True,name=name)
            except Exception:pass
        self.thermal_visible=True; self._redraw_objects(keep_thermal=True); self.plotter.render()

    def _restore_current_thermal(self):
        if self.current_thermal is not None:self._display_thermal(self.current_thermal,True)

    def _run_thermal(self):
        case=self.state.current_case
        if case is None or not case.anatomy.has_mask("csf"):
            QMessageBox.warning(self,"No patient anatomy","Process an MRI first."); return
        try:
            power,sim,sink=self._thermal_params()
            target=AnatomicalTarget(self.target_name.text(),Point3D(*self.target_point),"patient","manual")
            litt=LITTPlan("GUI-LITT-001",Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)),LITTParameters(float(self.active.text()),float(self.radius.text())))
            compat=CSFCompartment(center=Point3D(*self.target_point),radii=(18,18,18),resolution=48)
            ev=evaluate_litt_trajectory(litt,compat,target=target,seeg_electrodes=[self.seeg_object] if self.seeg_object is not None else [],uncertainty_enabled=True,patient_csf_mask=case.anatomy.get_mask("csf"))
            thermal=calculate_thermal_field(AblationGeometry(litt.trajectory,litt.parameters.active_length,litt.parameters.radius),compat,patient_csf_mask=case.anatomy.get_mask("csf"),grid_resolution=40,margin=12.,laser_power_watts=power,simulation_time_seconds=sim,time_step_seconds=.25,csf_sink_strength=sink)
            self.litt_object=litt; self.last_evaluation=ev; self.current_thermal=thermal
            try:case.add_litt_plan(litt)
            except Exception:pass
            self._display_evaluation(ev,thermal); self._display_thermal(thermal,False); self.statusBar().showMessage("ThermoSEEG full thermal model completed.")
        except Exception as exc:QMessageBox.critical(self,"Thermal analysis failed",str(exc))

    def _display_evaluation(self,ev,thermal=None):
        t=thermal or ev.thermal
        if t is None:return
        exp=ev.csf_exposure_fraction; peak=t.maximum_temperature; base=getattr(t,"baseline_temperature_celsius",37.)
        self.pie.set_exposure(exp)
        self.metrics.setText(f"TARGET\nDistance: {ev.target.target_distance:.2f} mm\n\nCSF\nExposure: {exp*100:.2f}%\nMinimum distance: {ev.minimum_csf_distance:.2f} mm\n\nTHERMAL\nPeak ΔT: {peak-base:.2f} °C\nPeak with CSF: {peak:.2f} °C\nPeak without CSF: {t.maximum_temperature_without_csf:.2f} °C\nCSF heat-sink reduction: {t.maximum_csf_temperature_reduction:.2f} °C\n\nDOSE\nWith CSF: {t.maximum_thermal_dose:.3f}\nWithout CSF: {t.maximum_thermal_dose_without_csf:.3f}\nDose reduction: {t.csf_thermal_dose_reduction:.3f}\n\nUNCERTAINTY\nTrajectories: {ev.uncertainty.number_of_trajectories if ev.uncertainty else 0}")

    def _capture_candidate(self):
        if self.state.current_case is None:return
        try:
            exp=self._current_exposure(); td=float(np.linalg.norm(self.litt_target-self.target_point)); sc=_distance_between_segments(self.litt_entry,self.litt_target,self.seeg_entry,self.seeg_target); tc=max(0.,_distance_point_to_segment(self.test_point,self.litt_entry,self.litt_target)-self.test_radius_value)
            r={"id":len(self.candidate_records)+1,"offset":float(self.offset.value()),"exposure":exp,"target":td,"seeg":sc,"test":tc}
            if self.current_thermal is not None:r["peak"]=float(self.current_thermal.maximum_temperature); r["reduction"]=float(self.current_thermal.maximum_csf_temperature_reduction)
            self.candidate_records.append(r); self._refresh_candidates(); self.statusBar().showMessage(f"Captured trajectory candidate {r['id']}.")
        except Exception as exc:self.statusBar().showMessage(f"Candidate capture unavailable: {exc}")

    def _current_exposure(self):
        case=self.state.current_case
        if case is None or not case.anatomy.has_mask("csf"):return 0.
        ab=AblationGeometry(Trajectory(Point3D(*self.litt_entry),Point3D(*self.litt_target)),float(self.active.text()),float(self.radius.text()))
        return float(calculate_mask_csf_exposure(ab,case.anatomy.get_mask("csf")).exposure_fraction)

    def _refresh_candidates(self):
        if not self.candidate_records:self.candidate_label.setText("No captured candidates."); return
        lines=["CAPTURED TRAJECTORIES"]
        for r in self.candidate_records[-8:]:
            th=f" | peak {r['peak']:.1f}°C | ΔCSF {r['reduction']:.2f}°C" if "peak" in r else ""
            lines.append(f"#{r['id']}  offset {r['offset']:.0f} mm | CSF {r['exposure']*100:.1f}% | target {r['target']:.1f} mm | SEEG {r['seeg']:.1f} mm{th}")
        self.candidate_label.setText("\n".join(lines))

    def _enable_target_picking(self):
        try:
            self.plotter.enable_surface_point_picking(callback=self._picked_target,show_message=True,show_point=True,clear_on_no_selection=False)
            self.statusBar().showMessage("Click the brain surface to place the target.")
        except Exception as exc:QMessageBox.warning(self,"Picking unavailable",str(exc))

    def _picked_target(self,point):
        p=np.asarray(point,float); self.target_point=p
        self._set_fields(p,self.tx,self.ty,self.tz); self._set_fields(p,self.ltargetx,self.ltargety,self.ltargetz); self._set_fields(p,self.etx,self.ety,self.etz)
        self._redraw_objects(); self._live_metrics(); self._schedule_thermal_preview(); self.statusBar().showMessage(f"Target placed at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")

    def _reset_view(self):self.plotter.reset_camera(); self.plotter.render()
    def _view_3d(self):self.plotter.reset_camera(); self.plotter.render()
    def _view_axial(self):self.plotter.view_xy(); self.plotter.render()
    def _view_coronal(self):self.plotter.view_xz(); self.plotter.render()
    def _view_sagittal(self):self.plotter.view_yz(); self.plotter.render()
    def closeEvent(self,event):
        try:self._live_timer.stop(); self._thermal_timer.stop(); self.plotter.close()
        except Exception:pass
        event.accept()

def launch_application():
    app=QApplication.instance() or QApplication([]); window=ThermoSEEGMainWindow(); window.show(); app.exec()

if __name__=="__main__":launch_application()
