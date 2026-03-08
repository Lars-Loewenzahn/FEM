import os
import json
import turtle
import sys
import math

BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODEL_PATH = os.path.join(TEMP_DIR, "model.json")
RESULTS_PATH = os.path.join(TEMP_DIR, "results.json")
PIXELS_PER_METER = 50.0

screen = None
t_orig = None
t_def = None
t_grid = None
t_axes = None
t_leg = None


def setup_screen():
    global screen, t_orig, t_def, t_grid, t_axes, t_leg
    screen = turtle.Screen()
    screen.title("FEM Postprocessor: q=Beenden")
    screen.setup(width=1000, height=800)
    screen.tracer(0, 0)

    t_orig = turtle.Turtle(visible=False)
    t_orig.penup()
    t_orig.speed(0)

    t_def = turtle.Turtle(visible=False)
    t_def.penup()
    t_def.speed(0)
    
    t_grid = turtle.Turtle(visible=False)
    t_grid.penup()
    t_grid.speed(0)

    t_axes = turtle.Turtle(visible=False)
    t_axes.penup()
    t_axes.speed(0)

    t_leg = turtle.Turtle(visible=False)
    t_leg.penup()
    t_leg.speed(0)


def stress_to_color(stress, smin, smax):
    # Map stress to color.
    # Negative: blue (min) -> green (0). Positive: green (0) -> red (max).
    # Purely positive: blue (min) -> red (max). Purely negative: blue (min) -> green (max=0).
    if not (isinstance(smin, (int, float)) and isinstance(smax, (int, float))):
        return (0.5, 0.5, 0.5)
    if abs(smax - smin) <= 1e-12:
        return (0.5, 0.5, 0.5)

    if smin >= 0.0:
        # All non-negative: blue -> red
        t = (stress - smin) / (smax - smin)
        t = max(0.0, min(1.0, t))
        return (t, 0.0, 1.0 - t)

    if smax <= 0.0:
        # All non-positive: blue -> green (up to 0)
        # Map smin..0 to 0..1
        rng = abs(smin) if abs(smin) > 1e-12 else 1.0
        t = (stress - smin) / rng
        t = max(0.0, min(1.0, t))
        # t=0 -> blue(0,0,1); t=1 -> green(0,1,0)
        return (0.0, t, 1.0 - t)

    # Mixed sign: piecewise mapping around 0
    if stress >= 0.0:
        t = stress / smax if abs(smax) > 1e-12 else 0.0
        t = max(0.0, min(1.0, t))
        # 0 (green) -> max (red)
        return (t, 1.0 - t, 0.0)
    else:
        t = stress / smin if abs(smin) > 1e-12 else 0.0  # stress and smin are negative
        t = max(0.0, min(1.0, t))
        # min (blue) -> 0 (green)
        return (0.0, 1.0 - t, t)


def m2px(v):
    return v * PIXELS_PER_METER


def draw_grid_and_axes():
    w = screen.window_width()
    h = screen.window_height()
    x_min = -w // 2
    x_max = w // 2
    y_min = -h // 2
    y_max = h // 2
    s = PIXELS_PER_METER

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
    t_axes.pencolor(0.85, 0.85, 0.85); t_axes.pensize(2)

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


def fmt_val(v):
    try:
        return f"{v:.3g}"
    except Exception:
        return str(v)


def draw_legend(smin, smax):
    # Vertical color bar at right side with labels at min, 0, max
    w = screen.window_width()
    h = screen.window_height()
    x_right = w // 2 - 30
    bar_w = 20
    bar_h = 220
    y_top = h // 2 - 40
    y_bot = y_top - bar_h

    smin_ext = min(smin, 0.0)
    smax_ext = max(smax, 0.0)

    steps = max(100, bar_h)
    t_leg.clear()
    for i in range(steps + 1):
        f = i / steps
        val = smin_ext + f * (smax_ext - smin_ext)
        col = stress_to_color(val, smin, smax)
        t_leg.pencolor(col)
        y = y_bot + f * bar_h
        t_leg.penup(); t_leg.goto(x_right - bar_w, y); t_leg.pendown(); t_leg.goto(x_right, y); t_leg.penup()

    # Box
    t_leg.pencolor("black")
    t_leg.penup(); t_leg.goto(x_right - bar_w, y_bot)
    t_leg.pendown(); t_leg.goto(x_right - bar_w, y_top)
    t_leg.goto(x_right, y_top)
    t_leg.goto(x_right, y_bot)
    t_leg.goto(x_right - bar_w, y_bot)
    t_leg.penup()

    # Labels
    t_leg.goto(x_right - bar_w - 6, y_top)
    t_leg.write(fmt_val(smax), align="right", font=("Arial", 10, "normal"))
    t_leg.goto(x_right - bar_w - 6, y_bot)
    t_leg.write(fmt_val(smin), align="right", font=("Arial", 10, "normal"))

    # Zero tick if within range
    if smin_ext < 0.0 or smax_ext > 0.0:
        rng = (smax_ext - smin_ext)
        if abs(rng) > 1e-12:
            f0 = (0.0 - smin_ext) / rng
            y0 = y_bot + f0 * bar_h
            t_leg.penup(); t_leg.goto(x_right - bar_w - 5, y0)
            t_leg.pendown(); t_leg.goto(x_right, y0)
            t_leg.penup(); t_leg.goto(x_right - bar_w - 6, y0)
            t_leg.write("0", align="right", font=("Arial", 10, "normal"))


def run_postprocessor(scale=None):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {MODEL_PATH}")
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {RESULTS_PATH}")

    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = json.load(f)
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    nodes = model["nodes"]
    elements = model["elements"]
    node_disp = {n["id"]: (0.0, 0.0) for n in nodes}
    for rn in results.get("nodes", []):
        node_disp[rn["id"]] = (rn.get("ux", 0.0), rn.get("uy", 0.0))
    elem_stress = {e["id"]: 0.0 for e in elements}
    for re in results.get("elements", []):
        elem_stress[re["id"]] = re.get("stress", 0.0)

    s_vals = list(elem_stress.values())
    smin = min(s_vals) if s_vals else 0.0
    smax = max(s_vals) if s_vals else 0.0

    setup_screen()

    s_input = None
    if scale is None:
        s_input = turtle.textinput("Deformationsfaktor", "Skalierung der Verschiebungen (z.B. 100)")
        try:
            scale = float(s_input) if s_input not in (None, "") else 1.0
        except Exception:
            scale = 1.0
    
    draw_grid_and_axes()
    t_orig.pencolor(0.7, 0.7, 0.7)
    t_orig.pensize(2)
    for e in elements:
        n1 = next(n for n in nodes if n["id"] == e["n1"])
        n2 = next(n for n in nodes if n["id"] == e["n2"])
        t_orig.penup(); t_orig.goto(m2px(n1["x"]), m2px(n1["y"]))
        t_orig.pendown(); t_orig.goto(m2px(n2["x"]), m2px(n2["y"])); t_orig.penup()

    t_def.pensize(3)
    for e in elements:
        n1 = next(n for n in nodes if n["id"] == e["n1"])
        n2 = next(n for n in nodes if n["id"] == e["n2"])
        ux1, uy1 = node_disp.get(n1["id"], (0.0, 0.0))
        ux2, uy2 = node_disp.get(n2["id"], (0.0, 0.0))
        x1d = m2px(n1["x"] + scale * ux1)
        y1d = m2px(n1["y"] + scale * uy1)
        x2d = m2px(n2["x"] + scale * ux2)
        y2d = m2px(n2["y"] + scale * uy2)
        col = stress_to_color(elem_stress.get(e["id"], 0.0), smin, smax)
        t_def.pencolor(col)
        t_def.penup(); t_def.goto(x1d, y1d)
        t_def.pendown(); t_def.goto(x2d, y2d); t_def.penup()

    draw_legend(smin, smax)
    screen.update()
    def _quit():
        try:
            screen.bye()
        except Exception:
            pass
    screen.listen(); screen.onkey(_quit, "q")
    turtle.done()


if __name__ == "__main__":
    scale = None
    if len(sys.argv) > 1:
        try:
            scale = float(sys.argv[1])
        except Exception:
            scale = None
    run_postprocessor(scale)

