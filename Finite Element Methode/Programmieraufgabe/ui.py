import os
import sys
import json

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    QT_LIB = "PyQt5"
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        QT_LIB = "PySide6"
    except ImportError as e:
        raise ImportError(
            "PyQt5 oder PySide6 ist erforderlich. Installiere z.B.: pip install PyQt5"
        ) from e


BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODEL_PATH = os.path.join(TEMP_DIR, "model.json")
RESULTS_PATH = os.path.join(TEMP_DIR, "results.json")
PREPROCESSOR = os.path.join(BASE_DIR, "preprocessor.py")
SOLVER = os.path.join(BASE_DIR, "solver.py")
POSTPROCESSOR = os.path.join(BASE_DIR, "postprocessor.py")


def read_model_defaults():
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        defs = data.get("defaults", {})
        return defs.get("E"), defs.get("A")
    except Exception:
        return None, None


def write_model_defaults(E, A, apply_to_elements=False):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelldatei fehlt. Bitte erst den Preprocessor ausführen.")
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("defaults", {})
    data["defaults"]["E"] = float(E)
    data["defaults"]["A"] = float(A)
    if apply_to_elements:
        for e in data.get("elements", []):
            e["E"] = float(E)
            e["A"] = float(A)
    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEM 2D Truss UI")
        self.resize(800, 600)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Modell
        g_model = QtWidgets.QGroupBox("Modell")
        fm = QtWidgets.QFormLayout()
        self.lbl_model = QtWidgets.QLabel(MODEL_PATH)
        fm.addRow("Modelldatei:", self.lbl_model)
        row = QtWidgets.QHBoxLayout()
        self.btn_new = QtWidgets.QPushButton("Neu")
        self.btn_new.clicked.connect(self.on_new_model)
        self.btn_load_model = QtWidgets.QPushButton("Öffnen…")
        self.btn_load_model.clicked.connect(self.on_load_model_file)
        self.btn_save_as = QtWidgets.QPushButton("Speichern unter…")
        self.btn_save_as.clicked.connect(self.on_save_model_as)
        self.btn_pre = QtWidgets.QPushButton("Preprocessor starten")
        self.btn_pre.clicked.connect(self.on_run_preprocessor)
        for b in (self.btn_new, self.btn_load_model, self.btn_save_as, self.btn_pre):
            row.addWidget(b)
        fm.addRow(row)
        g_model.setLayout(fm)
        layout.addWidget(g_model)

        # Parameter
        g_params = QtWidgets.QGroupBox("Parameter (Defaults)")
        fp = QtWidgets.QFormLayout()
        self.edit_E = QtWidgets.QLineEdit()
        self.edit_A = QtWidgets.QLineEdit()
        self.cb_apply = QtWidgets.QCheckBox("E/A auf alle vorhandenen Stäbe anwenden")
        self.btn_save_defaults = QtWidgets.QPushButton("Defaults speichern")
        self.btn_save_defaults.clicked.connect(self.on_save_defaults)
        fp.addRow("E [Pa]", self.edit_E)
        fp.addRow("A [m²]", self.edit_A)
        fp.addRow(self.cb_apply)
        fp.addRow(self.btn_save_defaults)
        g_params.setLayout(fp)
        layout.addWidget(g_params)

        # Solver
        g_solver = QtWidgets.QGroupBox("Solver")
        vs = QtWidgets.QVBoxLayout()
        self.btn_solver = QtWidgets.QPushButton("Solver starten")
        self.btn_solver.clicked.connect(self.on_run_solver)
        self.lbl_status = QtWidgets.QLabel("Bereit.")
        self.txt_log = QtWidgets.QTextEdit(readOnly=True)
        vs.addWidget(self.btn_solver)
        vs.addWidget(self.lbl_status)
        vs.addWidget(self.txt_log)
        g_solver.setLayout(vs)
        layout.addWidget(g_solver)

        # Postprocessor
        g_post = QtWidgets.QGroupBox("Postprocessor")
        fp2 = QtWidgets.QFormLayout()
        self.edit_scale = QtWidgets.QLineEdit("100")
        self.btn_post = QtWidgets.QPushButton("Postprocessor starten")
        self.btn_post.clicked.connect(self.on_run_postprocessor)
        fp2.addRow("Deformationsfaktor", self.edit_scale)
        fp2.addRow(self.btn_post)
        g_post.setLayout(fp2)
        layout.addWidget(g_post)

        g_res = QtWidgets.QGroupBox("Ergebnisse")
        vr = QtWidgets.QVBoxLayout()
        self.btn_disp = QtWidgets.QPushButton("Verschiebungen anzeigen")
        self.btn_disp.clicked.connect(self.on_show_displacements)
        self.btn_strain = QtWidgets.QPushButton("Verzerrungen anzeigen")
        self.btn_strain.clicked.connect(self.on_show_strains)
        self.btn_stress = QtWidgets.QPushButton("Spannungen anzeigen")
        self.btn_stress.clicked.connect(self.on_show_stresses)
        self.btn_reac = QtWidgets.QPushButton("Auflagerreaktionen anzeigen")
        self.btn_reac.clicked.connect(self.on_show_reactions)
        vr.addWidget(self.btn_disp)
        vr.addWidget(self.btn_strain)
        vr.addWidget(self.btn_stress)
        vr.addWidget(self.btn_reac)
        g_res.setLayout(vr)
        layout.addWidget(g_res)

        # Alles nacheinander
        self.btn_all = QtWidgets.QPushButton("Alles nacheinander (Pre→Solve→Post)")
        self.btn_all.clicked.connect(self.on_run_all)
        layout.addWidget(self.btn_all)

        # Statusbar
        self.statusBar().showMessage("Bereit")

        # Processes
        self.preproc_proc = None
        self.solver_proc = None
        self.postproc_proc = None
        self.chain_all = False

        # Initialwerte
        self.load_defaults_into_fields()
        self.refresh_ui_state()

    def load_defaults_into_fields(self):
        E, A = read_model_defaults()
        if E is None:
            E = 210e9
        if A is None:
            A = 1e-4
        self.edit_E.setText(str(E))
        self.edit_A.setText(str(A))

    def refresh_ui_state(self):
        exists_model = os.path.exists(MODEL_PATH)
        suffix = "(vorhanden)" if exists_model else "(fehlt)"
        self.lbl_model.setText(f"{MODEL_PATH} {suffix}")
        self.btn_solver.setEnabled(exists_model)
        has_results = os.path.exists(RESULTS_PATH)
        self.btn_post.setEnabled(has_results)
        if hasattr(self, 'btn_disp'):
            self.btn_disp.setEnabled(has_results)
        if hasattr(self, 'btn_strain'):
            self.btn_strain.setEnabled(has_results)
        if hasattr(self, 'btn_stress'):
            self.btn_stress.setEnabled(has_results)
        if hasattr(self, 'btn_reac'):
            self.btn_reac.setEnabled(has_results)

    def on_run_preprocessor(self):
        self.chain_all = False
        self.start_preprocessor()

    def start_preprocessor(self):
        if self.preproc_proc is None:
            self.preproc_proc = QtCore.QProcess(self)
            self.preproc_proc.finished.connect(self.on_preproc_finished)
            self.preproc_proc.errorOccurred.connect(self.on_proc_error)
        program = sys.executable
        args = [PREPROCESSOR]
        if os.path.exists(MODEL_PATH):
            args.append(MODEL_PATH)
        self.preproc_proc.start(program, args)
        if not self.preproc_proc.waitForStarted(1000):
            QtWidgets.QMessageBox.warning(self, "Fehler", "Preprocessor konnte nicht gestartet werden.")
        else:
            self.statusBar().showMessage("Preprocessor gestartet …")

    def on_load_model_file(self):
        try:
            dlg = QtWidgets.QFileDialog(self)
            dlg.setWindowTitle("Modell öffnen")
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            dlg.setNameFilters(["Modelldateien (*.json)", "Alle Dateien (*)"])
            if dlg.exec_() if hasattr(dlg, 'exec_') else dlg.exec():
                selected = dlg.selectedFiles()
                if selected:
                    src = selected[0]
                    self.load_model_file_to_temp(src)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Laden", f"Fehler beim Laden: {e}")

    def on_new_model(self):
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            # Basisgerüst leeres Modell
            E = float(self.edit_E.text()) if self.edit_E.text().strip() else 210e9
            A = float(self.edit_A.text()) if self.edit_A.text().strip() else 1e-4
            data = {
                "nodes": [],
                "elements": [],
                "supports": [],
                "loads": [],
                "defaults": {"E": E, "A": A},
            }
            with open(MODEL_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage("Neues Modell erstellt")
            self.refresh_ui_state()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Neu", f"Fehler beim Erstellen: {e}")

    def on_save_model_as(self):
        if not os.path.exists(MODEL_PATH):
            QtWidgets.QMessageBox.information(self, "Hinweis", "Kein aktuelles Modell vorhanden.")
            return
        try:
            with open(MODEL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Speichern", f"Ungültiges aktuelles Modell: {e}")
            return
        try:
            dlg = QtWidgets.QFileDialog(self)
            dlg.setWindowTitle("Modell speichern unter…")
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            dlg.setNameFilters(["Modelldateien (*.json)", "Alle Dateien (*)"])
            if dlg.exec_() if hasattr(dlg, 'exec_') else dlg.exec():
                selected = dlg.selectedFiles()
                if selected:
                    dst = selected[0]
                    with open(dst, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    self.statusBar().showMessage(f"Modell gespeichert: {dst}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Speichern", f"Fehler beim Speichern: {e}")

    def validate_model_dict(self, data: dict) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            # Minimalstruktur prüfen
            for key in ("nodes", "elements", "supports", "loads"):
                if key not in data:
                    return False
            if not isinstance(data["nodes"], list):
                return False
            if not isinstance(data["elements"], list):
                return False
            if not isinstance(data["supports"], list):
                return False
            if not isinstance(data["loads"], list):
                return False
            return True
        except Exception:
            return False

    def load_model_file_to_temp(self, src_path: str):
        os.makedirs(TEMP_DIR, exist_ok=True)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not self.validate_model_dict(data):
                QtWidgets.QMessageBox.warning(self, "Laden", "Ausgewählte Datei ist kein gültiges Modell (nodes/elements/supports/loads fehlen).")
                return
            with open(MODEL_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Modell geladen aus: {src_path}")
            self.load_defaults_into_fields()
            self.refresh_ui_state()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Laden", f"Ungültige Modelldatei: {e}")

    def on_preproc_finished(self):
        self.statusBar().showMessage("Preprocessor beendet.")
        self.preproc_proc = None
        # Nach Beenden Defaults neu laden
        self.load_defaults_into_fields()
        self.refresh_ui_state()
        if self.chain_all:
            self.on_run_solver()

    def on_save_defaults(self):
        try:
            E = float(self.edit_E.text())
            A = float(self.edit_A.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Eingabe", "Bitte gültige Zahlen für E und A eingeben.")
            return
        try:
            write_model_defaults(E, A, apply_to_elements=self.cb_apply.isChecked())
            self.statusBar().showMessage("Defaults gespeichert")
        except FileNotFoundError as e:
            QtWidgets.QMessageBox.information(self, "Hinweis", str(e))
        self.refresh_ui_state()

    def on_run_solver(self):
        if not os.path.exists(MODEL_PATH):
            QtWidgets.QMessageBox.information(self, "Hinweis", "Modelldatei fehlt. Bitte zuerst Preprocessor ausführen.")
            return
        # Vor dem Rechnen Defaults (und ggf. Stäbe) aktualisieren
        try:
            E = float(self.edit_E.text())
            A = float(self.edit_A.text())
            write_model_defaults(E, A, apply_to_elements=self.cb_apply.isChecked())
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Eingabe", f"Fehler bei E/A: {e}")
            return

        if self.solver_proc is None:
            self.solver_proc = QtCore.QProcess(self)
            self.solver_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.solver_proc.readyReadStandardOutput.connect(self.on_solver_output)
            self.solver_proc.finished.connect(self.on_solver_finished)
            self.solver_proc.errorOccurred.connect(self.on_proc_error)
        self.btn_solver.setEnabled(False)
        self.lbl_status.setText("Solver läuft …")
        self.txt_log.clear()
        self.solver_proc.start(sys.executable, [SOLVER])
        self.statusBar().showMessage("Solver gestartet …")

    def on_solver_output(self):
        if self.solver_proc is None:
            return
        data = self.solver_proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.txt_log.moveCursor(QtGui.QTextCursor.End)
            self.txt_log.insertPlainText(data)
            self.txt_log.moveCursor(QtGui.QTextCursor.End)

    def on_solver_finished(self):
        self.lbl_status.setText("Solver fertig.")
        self.btn_solver.setEnabled(True)
        self.solver_proc = None
        self.refresh_ui_state()
        if self.chain_all:
            self.on_run_postprocessor()

    def on_run_postprocessor(self):
        scale = None
        text = self.edit_scale.text().strip()
        if text:
            try:
                scale = float(text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Eingabe", "Bitte eine gültige Zahl für den Deformationsfaktor eingeben.")
                return
        if self.postproc_proc is None:
            self.postproc_proc = QtCore.QProcess(self)
            self.postproc_proc.finished.connect(self.on_postproc_finished)
            self.postproc_proc.errorOccurred.connect(self.on_proc_error)
        args = [POSTPROCESSOR]
        if scale is not None:
            args.append(str(scale))
        self.postproc_proc.start(sys.executable, args)
        self.statusBar().showMessage("Postprocessor gestartet …")

    def on_postproc_finished(self):
        self.statusBar().showMessage("Postprocessor beendet.")
        self.postproc_proc = None
        self.chain_all = False

    def on_run_all(self):
        self.chain_all = True
        self.start_preprocessor()

    def on_proc_error(self, err):
        QtWidgets.QMessageBox.critical(self, "Prozessfehler", f"Fehlercode: {int(err)}")

    def show_table(self, title, headers, rows):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        v = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(len(rows), len(headers), dlg)
        table.setHorizontalHeaderLabels(headers)
        def fmt(x):
            try:
                return f"{x:.6g}"
            except Exception:
                return str(x)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(val) if isinstance(val, (str,)) else fmt(val))
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        v.addWidget(table)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        btns.clicked.connect(lambda _btn: dlg.reject())
        v.addWidget(btns)
        dlg.resize(600, 400)
        dlg.exec_() if hasattr(dlg, 'exec_') else dlg.exec()

    def on_show_displacements(self):
        data = read_results()
        if not data:
            QtWidgets.QMessageBox.information(self, "Hinweis", "Keine Ergebnisse gefunden.")
            return
        nodes = data.get("nodes", [])
        headers = ["Knoten", "ux [m]", "uy [m]"]
        rows = [[n.get("id", ""), n.get("ux", 0.0), n.get("uy", 0.0)] for n in nodes]
        self.show_table("Verschiebungen", headers, rows)

    def on_show_strains(self):
        data = read_results()
        if not data:
            QtWidgets.QMessageBox.information(self, "Hinweis", "Keine Ergebnisse gefunden.")
            return
        elems = data.get("elements", [])
        headers = ["Element", "Verzerrung [-]"]
        rows = [[e.get("id", ""), e.get("strain", 0.0)] for e in elems]
        self.show_table("Verzerrungen (lokal)", headers, rows)

    def on_show_stresses(self):
        data = read_results()
        if not data:
            QtWidgets.QMessageBox.information(self, "Hinweis", "Keine Ergebnisse gefunden.")
            return
        elems = data.get("elements", [])
        headers = ["Element", "Spannung [Pa]"]
        rows = [[e.get("id", ""), e.get("stress", 0.0)] for e in elems]
        self.show_table("Spannungen", headers, rows)

    def on_show_reactions(self):
        data = read_results()
        if not data:
            QtWidgets.QMessageBox.information(self, "Hinweis", "Keine Ergebnisse gefunden.")
            return
        reacs = data.get("reactions", [])
        headers = ["Knoten", "Rx [N]", "Ry [N]"]
        rows = [[r.get("node", ""), r.get("rx", 0.0), r.get("ry", 0.0)] for r in reacs]
        self.show_table("Auflagerreaktionen", headers, rows)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    exec_fn = getattr(app, "exec", None)
    if callable(exec_fn):
        sys.exit(exec_fn())
    else:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()

