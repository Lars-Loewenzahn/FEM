import os
import sys
import json
import math
import turtle
import tkinter as tk
import tkinter.filedialog as fd

BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODEL_PATH = os.path.join(TEMP_DIR, "model.json")

nodes = []
elements = []
supports = {}
loads = {}

next_node_id = 1
next_elem_id = 1

default_E = 210e9
default_A = 1e-4
PIXELS_PER_METER = 50.0
GRID_SPACING = PIXELS_PER_METER

screen = None
t_nodes = None
t_elems = None
t_sups = None
t_loads = None
t_grid = None
t_axes = None
t_hud = None
ui_panel = None
ui_buttons = {}
ui_default_bg = None
ui_param_label = None

current_mode = 'node'
pending_element_node = None


def ensure_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)


def find_nearest_node(x, y, thresh=20.0):
    if not nodes:
        return None
    best = None
    best_d2 = float('inf')
    for n in nodes:
        dx = n["x"] * PIXELS_PER_METER - x
        dy = n["y"] * PIXELS_PER_METER - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = n["id"]
    if math.sqrt(best_d2) > thresh:
        return None
    return best


def m2px(v):
    return v * PIXELS_PER_METER


def px2m(v):
    return v / PIXELS_PER_METER


def snap_to_grid(x, y):
    s = GRID_SPACING if GRID_SPACING > 0 else 50.0
    xs = round(x / s) * s
    ys = round(y / s) * s
    return xs, ys


def input_two_values(title, label1, default1, label2, default2):
    try:
        cv = screen.getcanvas()
        root = cv.winfo_toplevel()
    except Exception:
        root = None
    result = {"vals": (default1, default2)}
    win = tk.Toplevel(root) if root else tk.Tk()
    win.title(title)
    frm = tk.Frame(win)
    frm.pack(fill="both", expand=True, padx=10, pady=10)
    l1 = tk.Label(frm, text=label1)
    l1.grid(row=0, column=0, sticky="w", padx=5, pady=5)
    e1 = tk.Entry(frm)
    e1.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    e1.insert(0, str(default1))
    l2 = tk.Label(frm, text=label2)
    l2.grid(row=1, column=0, sticky="w", padx=5, pady=5)
    e2 = tk.Entry(frm)
    e2.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
    e2.insert(0, str(default2))
    frm.columnconfigure(1, weight=1)
    btns = tk.Frame(frm)
    btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=8)
    def on_ok():
        s1 = e1.get().strip()
        s2 = e2.get().strip()
        try:
            v1 = float(s1) if s1 != "" else float(default1)
        except Exception:
            v1 = float(default1)
        try:
            v2 = float(s2) if s2 != "" else float(default2)
        except Exception:
            v2 = float(default2)
        result["vals"] = (v1, v2)
        win.destroy()
    def on_cancel():
        result["vals"] = (float(default1), float(default2))
        win.destroy()
    b_ok = tk.Button(btns, text="OK", command=on_ok)
    b_ok.pack(side="right", padx=5)
    b_cancel = tk.Button(btns, text="Abbrechen", command=on_cancel)
    b_cancel.pack(side="right", padx=5)
    try:
        win.grab_set()
        if root:
            win.transient(root)
    except Exception:
        pass
    e1.focus_set()
    win.bind("<Return>", lambda _e: on_ok())
    win.bind("<Escape>", lambda _e: on_cancel())
    win.wait_window(win)
    return result["vals"]


def setup_screen():
    global screen, t_nodes, t_elems, t_sups, t_loads, t_grid, t_axes, t_hud
    screen = turtle.Screen()
    screen.title("FEM Preprocessor: n=Knoten, e=Element, s=Lager, l=Last, p=Parameter, w=Speichern, q=Beenden")
    screen.setup(width=1000, height=800)
    screen.tracer(0, 0)
    try:
        cv = screen.getcanvas()
        try:
            tl = cv.winfo_toplevel()
            tl.focus_force()
        except Exception:
            cv.focus_force()
    except Exception:
        pass

    t_nodes = turtle.Turtle(visible=False)
    t_nodes.penup()
    t_nodes.speed(0)

    t_elems = turtle.Turtle(visible=False)
    t_elems.penup()
    t_elems.speed(0)

    t_sups = turtle.Turtle(visible=False)
    t_sups.penup()
    t_sups.speed(0)

    t_loads = turtle.Turtle(visible=False)
    t_loads.penup()
    t_loads.speed(0)

    t_grid = turtle.Turtle(visible=False)
    t_grid.penup()
    t_grid.speed(0)

    t_axes = turtle.Turtle(visible=False)
    t_axes.penup()
    t_axes.speed(0)

    t_hud = turtle.Turtle(visible=False)
    t_hud.hideturtle()
    t_hud.penup()
    t_hud.speed(0)


def draw_support_symbol(x, y, ux, uy):
    size = 10
    t_sups.pensize(2)
    t_sups.pencolor("black")
    if ux:
        t_sups.penup(); t_sups.goto(x - size, y - size)
        t_sups.pendown(); t_sups.goto(x - size, y + size); t_sups.penup()
    if uy:
        t_sups.penup(); t_sups.goto(x - size, y - size)
        t_sups.pendown(); t_sups.goto(x + size, y - size); t_sups.penup()


def draw_arrow(x, y, fx, fy):
    if abs(fx) < 1e-12 and abs(fy) < 1e-12:
        return
    L = math.hypot(fx, fy)
    ux = fx / L
    uy = fy / L
    scale = 50.0
    x2 = x + scale * ux
    y2 = y + scale * uy
    t_loads.pensize(2)
    t_loads.pencolor("red")
    t_loads.penup(); t_loads.goto(x, y)
    t_loads.pendown(); t_loads.goto(x2, y2); t_loads.penup()
    ah = 10.0
    ang = math.atan2(uy, ux)
    left = (x2 - ah * math.cos(ang - math.pi / 6), y2 - ah * math.sin(ang - math.pi / 6))
    right = (x2 - ah * math.cos(ang + math.pi / 6), y2 - ah * math.sin(ang + math.pi / 6))
    t_loads.goto(x2, y2); t_loads.pendown(); t_loads.goto(left[0], left[1]); t_loads.penup()
    t_loads.goto(x2, y2); t_loads.pendown(); t_loads.goto(right[0], right[1]); t_loads.penup()


def draw_grid_and_axes():
    w = screen.window_width()
    h = screen.window_height()
    x_min = -w // 2
    x_max = w // 2
    y_min = -h // 2
    y_max = h // 2

    s = GRID_SPACING if GRID_SPACING > 0 else 50.0

    t_grid.clear()
    t_grid.pencolor(0.85, 0.85, 0.85)
    t_grid.pensize(1)

    x = math.floor(x_min / s) * s
    while x <= x_max:
        t_grid.penup(); t_grid.goto(x, y_min); t_grid.pendown(); t_grid.goto(x, y_max); t_grid.penup()
        x += s

    y = math.floor(y_min / s) * s
    while y <= y_max:
        t_grid.penup(); t_grid.goto(x_min, y); t_grid.pendown(); t_grid.goto(x_max, y); t_grid.penup()
        y += s

    t_axes.clear()
    t_axes.pencolor(0.85, 0.85, 0.85)
    t_axes.pensize(2)

    t_axes.penup(); t_axes.goto(x_min, 0); t_axes.pendown(); t_axes.goto(x_max, 0); t_axes.penup()
    t_axes.penup(); t_axes.goto(0, y_min); t_axes.pendown(); t_axes.goto(0, y_max); t_axes.penup()

    tick = 5
    x = math.ceil(x_min / s) * s
    while x <= x_max:
        t_axes.penup(); t_axes.goto(x, -tick); t_axes.pendown(); t_axes.goto(x, tick); t_axes.penup()
        x += s
    y = math.ceil(y_min / s) * s
    while y <= y_max:
        t_axes.penup(); t_axes.goto(-tick, y); t_axes.pendown(); t_axes.goto(tick, y); t_axes.penup()
        y += s


def draw_hud():
    # Show current mode in the top-left corner
    t_hud.clear()
    w = screen.window_width()
    h = screen.window_height()
    x_min = -w // 2
    y_max = h // 2
    t_hud.goto(x_min + 10, y_max - 30)
    t_hud.color("black")
    t_hud.write(f"Modus: {current_mode}", align="left", font=("Arial", 12, "normal"))
    t_hud.goto(x_min + 10, y_max - 55)
    t_hud.write("Tasten: n=Knoten, e=Element, s=Lager, l=Last, p=Param, w=Speichern, q=Beenden", align="left", font=("Arial", 10, "normal"))
    # Show current element parameters (defaults used for new elements)
    t_hud.goto(x_min + 10, y_max - 80)
    try:
        t_hud.write(f"Element-Parameter: E={default_E:.3e} Pa, A={default_A:.3e} m^2", align="left", font=("Arial", 10, "normal"))
    except Exception:
        t_hud.write("Element-Parameter: E/A", align="left", font=("Arial", 10, "normal"))
    # If user has picked the first node for a new element, show it
    try:
        if current_mode == 'element' and pending_element_node is not None:
            t_hud.goto(x_min + 10, y_max - 100)
            t_hud.write(f"Neues Element: Startknoten n1={pending_element_node}", align="left", font=("Arial", 10, "normal"))
    except Exception:
        pass


def draw_all():
    t_elems.clear(); t_nodes.clear(); t_sups.clear(); t_loads.clear()
    draw_grid_and_axes()
    t_elems.pencolor("black"); t_elems.pensize(2)
    for e in elements:
        n1 = next(n for n in nodes if n["id"] == e["n1"])
        n2 = next(n for n in nodes if n["id"] == e["n2"])
        t_elems.penup(); t_elems.goto(m2px(n1["x"]), m2px(n1["y"]))
        t_elems.pendown(); t_elems.goto(m2px(n2["x"]), m2px(n2["y"])); t_elems.penup()

    for n in nodes:
        t_nodes.penup(); t_nodes.goto(m2px(n["x"]), m2px(n["y"])); t_nodes.dot(12, "blue")

    for nid, (ux, uy) in supports.items():
        n = next((nn for nn in nodes if nn["id"] == nid), None)
        if n:
            draw_support_symbol(m2px(n["x"]), m2px(n["y"]), ux, uy)

    for nid, (fx, fy) in loads.items():
        n = next((nn for nn in nodes if nn["id"] == nid), None)
        if n:
            draw_arrow(m2px(n["x"]), m2px(n["y"]), fx, fy)

    draw_hud()
    update_buttons_highlight()
    screen.update()


def cycle_support(ux, uy):
    states = [
        (False, False),
        (True, True),
        (True, False),
        (False, True),
    ]
    curr = (ux, uy)
    idx = states.index(curr)
    return states[(idx + 1) % len(states)]


def on_click(x, y):
    global next_node_id, next_elem_id, pending_element_node
    if current_mode == 'node':
        xs, ys = snap_to_grid(x, y)
        xm = px2m(xs)
        ym = px2m(ys)
        xv_m, yv_m = input_two_values("Knoten Koordinaten [m]", "x [m]", xm, "y [m]", ym)
        nodes.append({"id": next_node_id, "x": xv_m, "y": yv_m})
        next_node_id += 1
        draw_all()
        return

    nid = find_nearest_node(x, y)
    if nid is None:
        return

    if current_mode == 'element':
        if pending_element_node is None:
            pending_element_node = nid
        else:
            if pending_element_node != nid:
                elements.append({
                    "id": next_elem_id,
                    "n1": pending_element_node,
                    "n2": nid,
                    "E": default_E,
                    "A": default_A,
                })
                next_elem_id += 1
            pending_element_node = None
        draw_all()
        return

    if current_mode == 'support':
        ux, uy = supports.get(nid, (False, False))
        supports[nid] = cycle_support(ux, uy)
        draw_all()
        return

    if current_mode == 'support_h':
        supports[nid] = (False, True)
        draw_all()
        return

    if current_mode == 'support_v':
        supports[nid] = (True, False)
        draw_all()
        return

    if current_mode == 'load':
        dfx, dfy = loads.get(nid, (0.0, 0.0))
        fx, fy = input_two_values("Last [N]", "Fx (N, +→)", dfx, "Fy (N, +↑)", dfy)
        if abs(fx) < 1e-12 and abs(fy) < 1e-12:
            if nid in loads:
                del loads[nid]
        else:
            loads[nid] = (fx, fy)
        draw_all()
        return


def load_model_from_json(data: dict):
    global nodes, elements, supports, loads, default_E, default_A, next_node_id, next_elem_id
    try:
        nodes = list(data.get("nodes", []))
        elements = list(data.get("elements", []))
        # supports: list of {node, ux, uy} -> dict nid: (ux, uy)
        supports_list = data.get("supports", [])
        supports.clear()
        for s in supports_list:
            nid = int(s.get("node"))
            ux = bool(s.get("ux", False))
            uy = bool(s.get("uy", False))
            supports[nid] = (ux, uy)
        # loads: list of {node, fx, fy} -> dict nid: (fx, fy)
        loads_list = data.get("loads", [])
        loads.clear()
        for ld in loads_list:
            nid = int(ld.get("node"))
            fx = float(ld.get("fx", 0.0))
            fy = float(ld.get("fy", 0.0))
            loads[nid] = (fx, fy)
        defs = data.get("defaults", {})
        if "E" in defs:
            default_E = float(defs.get("E", default_E))
        if "A" in defs:
            default_A = float(defs.get("A", default_A))
        # next ids
        max_nid = max((n.get("id", 0) for n in nodes), default=0)
        max_eid = max((e.get("id", 0) for e in elements), default=0)
        next_node_id = int(max_nid) + 1
        next_elem_id = int(max_eid) + 1
    except Exception:
        # keep previous state on error
        pass
    refresh_param_label()


def load_model(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        load_model_from_json(data)
        print(f"Modell geladen: {path}")
    except Exception as e:
        print(f"Konnte Modell nicht laden: {e}")


def open_model_dialog():
    try:
        path = fd.askopenfilename(title="Modelldatei wählen", filetypes=[("Modelldatei", "*.json"), ("Alle Dateien", "*.*")])
        if path:
            load_model(path)
            draw_all()
    except Exception:
        pass


def set_mode_node():
    global current_mode, pending_element_node
    current_mode = 'node'
    pending_element_node = None
    draw_all()


def set_mode_element():
    global current_mode, pending_element_node
    current_mode = 'element'
    pending_element_node = None
    draw_all()


def set_mode_support():
    global current_mode, pending_element_node
    current_mode = 'support'
    pending_element_node = None
    draw_all()


def set_mode_support_h():
    global current_mode, pending_element_node
    current_mode = 'support_h'
    pending_element_node = None
    draw_all()


def set_mode_support_v():
    global current_mode, pending_element_node
    current_mode = 'support_v'
    pending_element_node = None
    draw_all()


def set_mode_load():
    global current_mode, pending_element_node
    current_mode = 'load'
    pending_element_node = None
    draw_all()


def setup_controls():
    global ui_panel, ui_buttons, ui_default_bg, ui_param_label
    try:
        cv = screen.getcanvas()
        root = cv.winfo_toplevel()
    except Exception:
        return
    ui_panel = tk.Toplevel(root)
    ui_panel.title("Werkzeuge")
    ui_panel.geometry("600x300+20+20")
    ui_panel.resizable(False, False)

    frm = tk.Frame(ui_panel)
    frm.pack(fill="both", expand=True, padx=10, pady=10)

    # Row: Model actions (horizontal)
    row_actions = tk.Frame(frm)
    row_actions.pack(fill="x", pady=5)
    tk.Button(row_actions, text="Neu", width=12, command=new_model).pack(side="left", padx=3)
    tk.Button(row_actions, text="Öffnen…", width=12, command=open_model_dialog).pack(side="left", padx=3)
    tk.Button(row_actions, text="Speichern", width=12, command=save_model).pack(side="left", padx=3)
    tk.Button(row_actions, text="Speichern unter…", width=16, command=save_model_as).pack(side="left", padx=3)
    tk.Button(row_actions, text="Parameter (p)", width=14, command=set_parameters).pack(side="left", padx=3)
    tk.Button(row_actions, text="Beenden", width=12, command=save_and_quit).pack(side="left", padx=3)

    # Current parameters row
    row_params = tk.Frame(frm)
    row_params.pack(fill="x", pady=4)
    ui_param_label = tk.Label(row_params, text="", anchor="w")
    ui_param_label.pack(fill="x")

    def add_btn(text, cmd):
        b = tk.Button(frm, text=text, width=28, command=cmd)
        b.pack(fill="x", pady=3)
        return b

    ui_buttons["node"] = add_btn("Knoten (n / 1)", set_mode_node)
    ui_buttons["element"] = add_btn("Element (e / 2)", set_mode_element)
    ui_buttons["support"] = add_btn("Lager (s / 3)", set_mode_support)
    ui_buttons["support_h"] = add_btn("Loslager Horizontal", set_mode_support_h)
    ui_buttons["support_v"] = add_btn("Loslager Vertikal", set_mode_support_v)
    ui_buttons["load"] = add_btn("Last (l / 4)", set_mode_load)

    sep = tk.Frame(frm, height=8)
    sep.pack(fill="x")

    # Remember default bg from one button
    if ui_buttons and ui_default_bg is None:
        try:
            some_key = next(iter(ui_buttons))
            bg = ui_buttons[some_key]['bg']
            globals()['ui_default_bg'] = bg
        except Exception:
            globals()['ui_default_bg'] = None

    # Initialize parameter label shown in the panel
    refresh_param_label()


def update_buttons_highlight():
    if not ui_buttons:
        return
    for mode, btn in ui_buttons.items():
        try:
            if mode == current_mode:
                btn.configure(bg="lightgreen", relief="sunken")
            else:
                if ui_default_bg is not None:
                    btn.configure(bg=ui_default_bg)
                else:
                    btn.configure(bg="SystemButtonFace")
                btn.configure(relief="raised")
        except Exception:
            pass


def set_parameters():
    global default_E, default_A
    sE = turtle.textinput("Material E", f"E (Pa), aktuell {default_E}")
    sA = turtle.textinput("Querschnitt A", f"A (m^2), aktuell {default_A}")
    try:
        if sE not in (None, ""):
            default_E = float(sE)
        if sA not in (None, ""):
            default_A = float(sA)
    except Exception:
        pass
    refresh_param_label()
    draw_all()


def save_model():
    ensure_temp_dir()
    data = build_current_model_data()
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Modell gespeichert: {MODEL_PATH}")


def save_and_quit():
    save_model()
    try:
        screen.bye()
    except Exception:
        pass


def build_current_model_data() -> dict:
    return {
        "nodes": nodes,
        "elements": elements,
        "supports": [
            {"node": nid, "ux": ux, "uy": uy} for nid, (ux, uy) in supports.items()
        ],
        "loads": [
            {"node": nid, "fx": fx, "fy": fy} for nid, (fx, fy) in loads.items()
        ],
        "defaults": {"E": default_E, "A": default_A},
    }


def save_model_as():
    try:
        data = build_current_model_data()
        path = fd.asksaveasfilename(title="Modell speichern unter…", defaultextension=".json",
                                     filetypes=[("Modelldatei", "*.json"), ("Alle Dateien", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Modell gespeichert: {path}")
    except Exception:
        pass


def new_model():
    global nodes, elements, supports, loads, next_node_id, next_elem_id
    nodes = []
    elements = []
    supports = {}
    loads = {}
    next_node_id = 1
    next_elem_id = 1
    save_model()
    draw_all()
    refresh_param_label()


def refresh_param_label():
    try:
        if ui_param_label is not None:
            ui_param_label.configure(text=f"Aktuelle Element-Parameter: E={default_E:.3e} Pa, A={default_A:.3e} m^2")
    except Exception:
        pass


def run_preprocessor():
    setup_screen()
    setup_controls()
    # optional: Modelldatei als CLI-Argument
    try:
        if len(sys.argv) > 1:
            arg_path = sys.argv[1]
            if isinstance(arg_path, str) and os.path.exists(arg_path):
                load_model(arg_path)
    except Exception:
        pass
    draw_all()
    screen.onclick(on_click)
    screen.listen()
    # Bind both onkey and onkeypress to improve reliability across platforms
    for binder in (screen.onkey, screen.onkeypress):
        binder(set_mode_node, "n"); binder(set_mode_node, "N")
        binder(set_mode_element, "e"); binder(set_mode_element, "E")
        binder(set_mode_support, "s"); binder(set_mode_support, "S")
        binder(set_mode_load, "l"); binder(set_mode_load, "L")
        binder(set_parameters, "p"); binder(set_parameters, "P")
        binder(save_model, "w"); binder(save_model, "W")
        binder(save_and_quit, "q"); binder(save_and_quit, "Q")
        # Numeric shortcuts
        binder(set_mode_node, "1")
        binder(set_mode_element, "2")
        binder(set_mode_support, "3")
        binder(set_mode_load, "4")
    screen.listen()
    turtle.done()


if __name__ == "__main__":
    run_preprocessor()

