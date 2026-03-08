import random
import turtle
import time
import math
import copy
import os
import json


CAR_COLORS = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "yellow", "brown", "black"
]

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def function(self, x):
        return 1/(1+math.exp(-x))

    def activate(self, inputs):
        z = sum(x*w for x, w in zip(inputs, self.weights)) + self.bias
        return self.function(z)
            
class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.neurons = []

        for i in range(hidden_size):
            weights = []
            for i in range(hidden_size):
                weights.append(random.uniform(-1, 1))
                bias = random.uniform(-1, 1)
            self.neurons.append(Neuron(weights, bias))

        for i in range(output_size):
            weights = []
            for i in range(hidden_size):
                weights.append(random.uniform(-1, 1))
                bias = random.uniform(-1, 1)
            self.neurons.append(Neuron(weights, bias))
    
    def forward(self, inputs):
        hidden = [neuron.activate(inputs) for neuron in self.neurons[:self.hidden_size]]
        output = [neuron.activate(hidden) for neuron in self.neurons[self.hidden_size:]]
        return output

    def mutate(self, factor):
        for neuron in self.neurons:
            for i in range(len(neuron.weights)):
                if random.random() < 1:
                    neuron.weights[i] += random.uniform(-factor, factor)
            if random.random() < 1:
                neuron.bias += random.uniform(-factor, factor)

    def get_params(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "layers": [
                {"weights": list(neuron.weights), "bias": neuron.bias}
                for neuron in self.neurons
            ],
        }

    def set_params(self, params):
        try:
            if (
                params.get("input_size") != self.input_size or
                params.get("hidden_size") != self.hidden_size or
                params.get("output_size") != self.output_size
            ):
                return False
            layers = params.get("layers")
            if not isinstance(layers, list) or len(layers) != len(self.neurons):
                return False
            for n, p in zip(self.neurons, layers):
                w = p.get("weights")
                b = p.get("bias")
                if not isinstance(w, list) or b is None:
                    return False
                if len(w) != len(n.weights):
                    return False
                n.weights = list(w)
                n.bias = float(b)
            return True
        except Exception:
            return False



class Track:
    def __init__(self, cols=21, rows=15, cell=24, wall_thickness=3, margin=20, seed=None):
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell = int(cell)
        self.wall_thickness = int(wall_thickness)
        self.margin = int(margin)
        self.rng = random.Random(seed)
        self.grid = [[{"N": True, "S": True, "E": True, "W": True} for _ in range(self.cols)] for _ in range(self.rows)]
        self.generated = False

    def generate(self):
        visited = [[False] * self.cols for _ in range(self.rows)]
        stack = []
        r = self.rng.randrange(self.rows)
        c = self.rng.randrange(self.cols)
        visited[r][c] = True
        stack.append((r, c))
        while stack:
            cr, cc = stack[-1]
            neighbors = []
            if cr > 0 and not visited[cr - 1][cc]:
                neighbors.append((cr - 1, cc, "N"))
            if cr < self.rows - 1 and not visited[cr + 1][cc]:
                neighbors.append((cr + 1, cc, "S"))
            if cc > 0 and not visited[cr][cc - 1]:
                neighbors.append((cr, cc - 1, "W"))
            if cc < self.cols - 1 and not visited[cr][cc + 1]:
                neighbors.append((cr, cc + 1, "E"))
            if neighbors:
                nr, nc, direction = self.rng.choice(neighbors)
                self._remove_wall(cr, cc, nr, nc, direction)
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()
        self.grid[0][0]["W"] = False
        self.grid[self.rows - 1][self.cols - 1]["E"] = False
        self.generated = True

    def _remove_wall(self, r1, c1, r2, c2, direction):
        if direction == "N":
            self.grid[r1][c1]["N"] = False
            self.grid[r2][c2]["S"] = False
        elif direction == "S":
            self.grid[r1][c1]["S"] = False
            self.grid[r2][c2]["N"] = False
        elif direction == "W":
            self.grid[r1][c1]["W"] = False
            self.grid[r2][c2]["E"] = False
        elif direction == "E":
            self.grid[r1][c1]["E"] = False
            self.grid[r2][c2]["W"] = False

    def _cell_top_left(self, r, c):
        width = self.cols * self.cell
        height = self.rows * self.cell
        left = -width / 2
        top = height / 2
        x0 = left + c * self.cell
        y0 = top - r * self.cell
        return x0, y0

    def _draw_line(self, pen, x1, y1, x2, y2):
        pen.up()
        pen.goto(x1, y1)
        pen.down()
        pen.goto(x2, y2)

    def draw(self):
        if not self.generated:
            self.generate()
        screen = turtle.Screen()
        width_px = int(self.cols * self.cell + 2 * self.margin)
        height_px = int(self.rows * self.cell + 2 * self.margin)
        try:
            screen.setup(width=width_px, height=height_px)
        except turtle.Terminator:
            screen = turtle.Screen()
            screen.setup(width=width_px, height=height_px)
        screen.title("Turtle Track")
        screen.tracer(False)
        pen = turtle.Turtle(visible=False)
        pen.speed(0)
        pen.color("black")
        pen.width(self.wall_thickness)

        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = self._cell_top_left(r, c)
                if self.grid[r][c]["N"]:
                    self._draw_line(pen, x0, y0, x0 + self.cell, y0)
                if self.grid[r][c]["W"]:
                    self._draw_line(pen, x0, y0, x0, y0 - self.cell)

        r = self.rows - 1
        for c in range(self.cols):
            if self.grid[r][c]["S"]:
                x0, y0 = self._cell_top_left(r, c)
                self._draw_line(pen, x0, y0 - self.cell, x0 + self.cell, y0 - self.cell)

        c = self.cols - 1
        for r in range(self.rows):
            if self.grid[r][c]["E"]:
                x0, y0 = self._cell_top_left(r, c)
                self._draw_line(pen, x0 + self.cell, y0, x0 + self.cell, y0 - self.cell)

        screen.update()
        return screen, pen

    def run(self):
        self.generate()
        self.draw()
        turtle.done()


class Ray:
    def __init__(self, track, screen, x, y, heading_deg, debug_draw=False, color="yellow", width=1):
        self.track = track
        self.screen = screen       # übergeben, nicht neu erstellen
        self.x, self.y = float(x), float(y)
        self.heading = float(heading_deg) % 360.0
        self.debug_draw = debug_draw
        self.t = None
        if self.debug_draw:
            self.t = turtle.Turtle(visible=False)
            self.t.hideturtle()
            self.t.speed(0)
            self.t.color(color)
            self.t.width(width)
            self.t.penup()

    def _dims(self):
        width = self.track.cols * self.track.cell
        height = self.track.rows * self.track.cell
        left = -width / 2
        top = height / 2
        return left, top, self.track.cell, self.track.rows, self.track.cols

    def destroy(self):
        # Bei debug_draw optional aufräumen, sonst nichts tun
        if self.debug_draw and self.t:
            try:
                self.t.clear()
                self.t.hideturtle()
            except turtle.Terminator:
                pass

    def cast(self, max_iters=10000):
        # ... Distanz rein rechnerisch bestimmen (dein vorhandener Code bis end_x,end_y,length) ...
        left, top, cell, rows, cols = self._dims()
        dx = math.cos(math.radians(self.heading))
        dy = math.sin(math.radians(self.heading))
        x = self.x
        y = self.y
        r = int((top - y) // cell)
        c = int((x - left) // cell)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0.0
        length = 0.0
        end_x = x
        end_y = y
        eps = 1e-12
        it = 0
        while it < max_iters:
            it += 1
            if dx > 0:
                next_vx = left + (c + 1) * cell
                t_vert = (next_vx - x) / dx
            elif dx < 0:
                next_vx = left + c * cell
                t_vert = (next_vx - x) / dx
            else:
                t_vert = float("inf")
            if dy > 0:
                next_hy = top - r * cell
                t_horiz = (next_hy - y) / dy
            elif dy < 0:
                next_hy = top - (r + 1) * cell
                t_horiz = (next_hy - y) / dy
            else:
                t_horiz = float("inf")
            if t_vert <= eps and t_horiz <= eps:
                t_vert = t_horiz = 0.0
            if t_vert < t_horiz:
                step_t = t_vert
                x2 = x + dx * step_t
                y2 = y + dy * step_t
                wall = self.track.grid[r][c]["E"] if dx > 0 else self.track.grid[r][c]["W"]
                if wall:
                    half_t = self.track.wall_thickness / 2.0
                    denom = abs(dx) if abs(dx) > 1e-12 else float("inf")
                    t_contact = max(0.0, step_t - half_t / denom)
                    end_x, end_y = x + dx * t_contact, y + dy * t_contact
                    length += t_contact
                    break
                x, y = x2, y2
                length += max(0.0, step_t)
                c = c + 1 if dx > 0 else c - 1
                if c < 0 or c >= cols:
                    end_x, end_y = x, y
                    break
            elif t_horiz < t_vert:
                step_t = t_horiz
                x2 = x + dx * step_t
                y2 = y + dy * step_t
                wall = self.track.grid[r][c]["N"] if dy > 0 else self.track.grid[r][c]["S"]
                if wall:
                    half_t = self.track.wall_thickness / 2.0
                    denom = abs(dy) if abs(dy) > 1e-12 else float("inf")
                    t_contact = max(0.0, step_t - half_t / denom)
                    end_x, end_y = x + dx * t_contact, y + dy * t_contact
                    length += t_contact
                    break
                x, y = x2, y2
                length += max(0.0, step_t)
                r = r - 1 if dy > 0 else r + 1
                if r < 0 or r >= rows:
                    end_x, end_y = x, y
                    break
            else:
                step_t = t_vert
                x2 = x + dx * step_t
                y2 = y + dy * step_t
                wall_v = self.track.grid[r][c]["E"] if dx > 0 else self.track.grid[r][c]["W"]
                wall_h = self.track.grid[r][c]["N"] if dy > 0 else self.track.grid[r][c]["S"]
                if wall_v or wall_h:
                    half_t = self.track.wall_thickness / 2.0
                    adj_v = half_t / (abs(dx) if abs(dx) > 1e-12 else float("inf")) if wall_v else float("inf")
                    adj_h = half_t / (abs(dy) if abs(dy) > 1e-12 else float("inf")) if wall_h else float("inf")
                    reduce_t = min(adj_v, adj_h)
                    t_contact = max(0.0, step_t - reduce_t)
                    end_x, end_y = x + dx * t_contact, y + dy * t_contact
                    length += t_contact
                    break
                x, y = x2, y2
                length += max(0.0, step_t)
                c = c + 1 if dx > 0 else c - 1
                r = r - 1 if dy > 0 else r + 1
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    end_x, end_y = x, y
                    break
        # Zeichnen nur, wenn debug_draw aktiv ist
        if self.debug_draw and self.t:
            try:
                self.t.clear()
                self.t.goto(self.x, self.y)
                self.t.pendown()
                self.t.goto(end_x, end_y)
                self.t.penup()
            except turtle.Terminator:
                pass
        return float(length)

 
class Car:
    def __init__(self, track, color=None):
        self.track = track
        self.screen = turtle.Screen()
        self.t = turtle.Turtle()
        self.t.shape("turtle")
        if color is None:
            color = random.choice(CAR_COLORS)
        self.color = color
        self.t.color(self.color)
        self.t.penup()
        self.t.speed(0)

        cx, cy = self._cell_center(0, 0)
        self.x, self.y = cx, cy
        self.heading = 0.0
        self.speed = 0.0
        self.spawn_x, self.spawn_y, self.spawn_heading = self.x, self.y, self.heading

        # besuchte Zellen
        self.visited_cells = set()
        r0, c0 = self._current_cell()
        if r0 is not None:
            self.visited_cells.add((r0, c0))
            self.cell_r, self.cell_c = r0, c0
        else:
            self.cell_r = self.cell_c = -1
        self.left_turns = 0
        self.right_turns = 0
        self.collided_recently = False
        self.time_since_collision = 0.0
        self.max_distance_from_start = 0.0
        self.distance_travelled = 0.0
        self.accel = 100.0
        self.brake_accel = 100.0
        self.drag = 0.8
        self.turn_rate = 140.0
        self.max_speed = 260.0
        self.max_reverse = 140.0
        self.keys = {"w": False, "s": False, "a": False, "d": False}
        self.collisions = 0
        self.score = 0
        self.cell_r = 0
        self.cell_c = 0

        self.t.goto(self.x, self.y)
        self.t.setheading(self.heading)
        self.screen.update()

        self._bind_keys()
        self._last_time = time.perf_counter()
        self._running = False
        self.lidar_distances = []

        ###  AI   ###  AI ###  AI   ###  AI ###  AI   ###  AI ###  AI   ###  AI ###  AI   ###  AI 
        self.brain = Network(24, 24, 4) 

    def use_brain(self):
        features = self.lidar_distances + [
            self.speed / self.max_speed,
            (self.heading % 360) / 360.0, self.t.xcor()/2250, self.t.ycor()/2250
        ]
        ans = self.brain.forward(features)
        self.keys = {"w": False, "s": False, "a": False, "d": False}
        if ans[0] >= ans[1]:
            self.keys["w"] = True
        if ans[1] > ans[0] and ans[1] > 0.5:
            self.keys["s"] = True
    
        # Exklusive Wahl: Lenken
        if ans[2] >= ans[3] and ans[2] > 0.5:
            self.keys["a"] = True
        if ans[3] > ans[2] and ans[3] > 0.5:
            self.keys["d"] = True

    def mutate(self, factor):
        self.brain.mutate(factor)

    def clone(self):
        child = Car(self.track)          
        child.brain = copy.deepcopy(self.brain)  
        return child

         ###  AI   ###  AI ###  AI   ###  AI ###  AI   ###  AI ###  AI   ###  AI ###  AI   ###  AI 

    def _cell_center(self, r, c):
        x0, y0 = self.track._cell_top_left(r, c)
        return x0 + self.track.cell / 2, y0 - self.track.cell / 2

    def _dims(self):
        width = self.track.cols * self.track.cell
        height = self.track.rows * self.track.cell
        left = -width / 2
        top = height / 2
        return left, top, self.track.cell, self.track.rows, self.track.cols

    def _current_cell(self):
        left, top, cell, rows, cols = self._dims()
        r = int((top - self.y) // cell)
        c = int((self.x - left) // cell)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return None, None
        return r, c
    
    def replace_track(self, track):
        self.track = track
        self.spawn_x, self.spawn_y = self._cell_center(0, 0)
        self.spawn_heading = 0.0

    def _bind_keys(self):
        s = self.screen
        s.listen()
        s.onkeypress(lambda: self._set_key("w", True), "w")
        s.onkeypress(lambda: self._set_key("w", True), "W")
        s.onkeypress(lambda: self._set_key("s", True), "s")
        s.onkeypress(lambda: self._set_key("s", True), "S")
        s.onkeypress(lambda: self._set_key("a", True), "a")
        s.onkeypress(lambda: self._set_key("a", True), "A")
        s.onkeypress(lambda: self._set_key("d", True), "d")
        s.onkeypress(lambda: self._set_key("d", True), "D")
        try:
            s.onkeyrelease(lambda: self._set_key("w", False), "w")
            s.onkeyrelease(lambda: self._set_key("w", False), "W")
            s.onkeyrelease(lambda: self._set_key("s", False), "s")
            s.onkeyrelease(lambda: self._set_key("s", False), "S")
            s.onkeyrelease(lambda: self._set_key("a", False), "a")
            s.onkeyrelease(lambda: self._set_key("a", False), "A")
            s.onkeyrelease(lambda: self._set_key("d", False), "d")
            s.onkeyrelease(lambda: self._set_key("d", False), "D")
        except Exception:
            pass

    def _set_key(self, k, v):
        self.keys[k] = v


    def _cast_rays(self):
        self.lidar_distances = []
        for i in range(20):
            angle = self.heading + i * 18
            ray = Ray(self.track, self.screen,self.x, self.y, angle)
            length = ray.cast()
            self.lidar_distances.append(length/200)
            ray.destroy()
        
    def update(self, dt):
        prev_tx, prev_ty = self.t.xcor(), self.t.ycor()
        if getattr(self, "collided_recently", False):
            self.time_since_collision += dt
        self._update_physics(dt)
        self._apply_movement(dt)
        self._cast_rays()
        self.use_brain()
        self.t.setheading(self.heading)
        self.t.goto(self.x, self.y)
        # Travelled distance and max distance from start based on turtle coords
        cur_tx, cur_ty = self.t.xcor(), self.t.ycor()
        moved = math.hypot(cur_tx - prev_tx, cur_ty - prev_ty)
        self.distance_travelled += moved
        dist_start = math.hypot(cur_tx - self.spawn_x, cur_ty - self.spawn_y)
        if dist_start > self.max_distance_from_start:
            self.max_distance_from_start = dist_start

        # Unique cells based on turtle coords, after moving
        left, top, cell, rows, cols = self._dims()
        r = int((top - cur_ty) // cell)
        c = int((cur_tx - left) // cell)
        if 0 <= r < rows and 0 <= c < cols:
            if r != self.cell_r or c != self.cell_c:
                if (r, c) not in self.visited_cells:
                    if self.speed > 0:
                        self.score += 1 * len(self.visited_cells)
                    else:
                        self.score += 1
                    self.visited_cells.add((r, c))
                self.cell_r, self.cell_c = r, c
        if getattr(self, "collided_recently", False) and abs(self.speed) > 30.0:
            reward = max(0.0, 5.0 - 2.0 * self.time_since_collision)
            self.score += reward
            self.collided_recently = False
    
    def reset(self):
        # Logikzustand
        self.speed = 0.0
        self.collisions = 0
        self.score = 0
        self.keys = {"w": False, "s": False, "a": False, "d": False}
        self.lidar_distances = [0.0]*20

        # Pose zurücksetzen
        self.x, self.y = self.spawn_x, self.spawn_y
        self.heading = self.spawn_heading
        self.t.penup()
        self.t.goto(self.x, self.y)
        self.t.setheading(self.heading)
        self.left_turns = 0
        self.right_turns = 0
        self.collided_recently = False
        self.time_since_collision = 0.0
        self.max_distance_from_start = 0.0
        self.distance_travelled = 0.0
        self.visited_cells = set()
        r, c = self._current_cell()
        if r is not None:
            self.visited_cells.add((r, c))
            self.cell_r, self.cell_c = r, c
        else:
            self.cell_r = self.cell_c = -1


    def _update_physics(self, dt):
        prev_diff = (self.left_turns - self.right_turns) if hasattr(self, "left_turns") else 0
        if self.keys["a"] != self.keys["d"]:
            if self.keys["a"]:
                self.left_turns = getattr(self, "left_turns", 0) + 1
            else:
                self.right_turns = getattr(self, "right_turns", 0) + 1
            if abs(self.left_turns - self.right_turns) < abs(prev_diff):
                self.score += 0.5
        if self.keys["a"]:
            self.heading = (self.heading + self.turn_rate * dt) % 360.0
        if self.keys["d"]:
            self.heading = (self.heading - self.turn_rate * dt) % 360.0
        throttle = 1.0 if self.keys["w"] else 0.0
        brake = 1.0 if self.keys["s"] else 0.0
        net_accel = throttle * self.accel - brake * self.brake_accel - self.drag * self.speed
        self.speed += net_accel * dt
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_reverse:
            self.speed = -self.max_reverse

    def _apply_movement(self, dt):
        dx = math.cos(math.radians(self.heading)) * self.speed * dt
        dy = math.sin(math.radians(self.heading)) * self.speed * dt
        self._move_axis(dx, axis="x")
        self._move_axis(dy, axis="y")

    def _move_axis(self, d, axis="x"):
        if d == 0.0:
            return
        left, top, cell, rows, cols = self._dims()
        collided = False

        if axis == "x":
            old_x = self.x
            new_x = self.x + d
            r = int((top - self.y) // cell)
            if r < 0 or r >= rows:
                collided = True
                new_x = old_x
            else:
                c0 = int((old_x - left) // cell)
                half = self.track.wall_thickness / 2.0
                # Clamp against right wall edge inside current cell
                if d > 0:
                    if 0 <= c0 < cols:
                        bx = left + (c0 + 1) * cell  # gridline x between c0 and c0+1
                        if self.track.grid[r][c0]["E"]:
                            x_max = bx - half
                            if new_x > x_max:
                                new_x = x_max
                                collided = True
                    else:
                        collided = True
                        new_x = old_x
                # Clamp against left wall edge inside current cell
                elif d < 0:
                    if 0 <= c0 < cols:
                        bx = left + c0 * cell  # left gridline of cell c0
                        if self.track.grid[r][c0]["W"]:
                            x_min = bx + half
                            if new_x < x_min:
                                new_x = x_min
                                collided = True
                    else:
                        collided = True
                        new_x = old_x
            self.x = new_x
        else:
            old_y = self.y
            new_y = self.y + d
            c = int((self.x - left) // cell)
            if c < 0 or c >= cols:
                collided = True
                new_y = old_y
            else:
                r0 = int((top - old_y) // cell)
                half = self.track.wall_thickness / 2.0
                # Moving up (towards north wall)
                if d > 0:
                    if 0 <= r0 < rows:
                        by = top - r0 * cell  # north gridline of row r0
                        if self.track.grid[r0][c]["N"]:
                            y_max = by - half
                            if new_y > y_max:
                                new_y = y_max
                                collided = True
                    else:
                        collided = True
                        new_y = old_y
                # Moving down (towards south wall)
                elif d < 0:
                    if 0 <= r0 < rows:
                        by = top - (r0 + 1) * cell  # south gridline of row r0
                        if self.track.grid[r0][c]["S"]:
                            y_min = by + half
                            if new_y < y_min:
                                new_y = y_min
                                collided = True
                    else:
                        collided = True
                        new_y = old_y
            self.y = new_y

        if collided:
            self.collisions += 1
            #self.score -= 1*self.collisions
            self.speed = 0.0
            self.collided_recently = True
            self.time_since_collision = 0.0
    
    def destroy(self):
        try:
            self.t.clear()        # löscht evtl. gezeichnete Linien/Stempel
            self.t.hideturtle()   # blendet die Turtle-Form aus
        except turtle.Terminator:
            pass

# -------------------- Scoring-Strategien (Generation-time) --------------------
def score_max_distance_from_start(car):
    return car.max_distance_from_start

def score_fewest_collisions(car):
    # weniger Kollisionen => höherer Score
    return -car.collisions

def score_balanced_steering(car):
    total = car.left_turns + car.right_turns
    if total == 0:
        return 0.0
    balance = 1.0 - abs(car.left_turns - car.right_turns) / total  # 0..1
    # stärker gewichten, wenn viel gelenkt wurde
    return balance * total

def score_unique_cells(car):
    return float(len(car.visited_cells))

def score_progress_efficiency(car):
    # eigene Idee: viele neue Zellen, weit weg vom Start, wenige Kollisionen
    return 20 * len(car.visited_cells) + 1 * car.max_distance_from_start - 0.1 * car.collisions + 0.1*score_balanced_steering(car)

SCORERS = {
    "max_distance": score_max_distance_from_start,
    "fewest_collisions": score_fewest_collisions,
    "balanced_steering": score_balanced_steering,
    "unique_cells": score_unique_cells,
    "progress_efficiency": score_progress_efficiency,
}

# Wähle hier den aktiven Scorer (einfachen String ändern)
#"balanced_steering"   # oder "max_distance", "fewest_collisions", "unique_cells", "progress_efficiency"
ACTIVE_SCORER_KEY = "progress_efficiency"


BEST_BRAIN_PATH = os.path.join(os.path.dirname(__file__), "best_brain.json")

def save_best_brain(network, path=BEST_BRAIN_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(network.get_params(), f)
    except Exception:
        pass

def load_best_brain(path=BEST_BRAIN_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None



frame_counter = 0

if __name__ == "__main__":
    track = Track(cols=12, rows=6, cell=150, wall_thickness=20, seed=None)
    screen, maze_pen = track.draw()
    cars = [Car(track) for _ in range(30)]
    _loaded_params = load_best_brain()
    if _loaded_params:
        for _car in cars:
            _car.brain.set_params(_loaded_params)
    
    def next_generation():
            global cars, ACTIVE_SCORER_KEY, track, maze_pen, screen
            scorer = SCORERS.get(ACTIVE_SCORER_KEY, score_progress_efficiency)
            for c in cars:
                c.score = scorer(c)
            ranked = sorted(cars, key=lambda c: c.score, reverse=True)
            losers = ranked[15:]
            for loser in losers:
                loser.destroy()
            survivors = ranked[:15]
            cars = survivors[:]
            save_best_brain(cars[0].brain)
            print("High Scores:", cars[0].score)
            print("Low Scores:", cars[-1].score)
            print("Max Distance:", cars[0].max_distance_from_start)
            print("Collisions:", cars[0].collisions)
            print("Unique Cells:", len(cars[0].visited_cells))
            print("Turns:", cars[0].left_turns - cars[0].right_turns , "\n \n \n")

            for i in range(15):
                cars.append(survivors[i].clone())

            if maze_pen is not None:
                try:
                    maze_pen.clear()
                except Exception:
                    pass
            track = Track(cols=12, rows=6, cell=150, wall_thickness=20, seed=None)
            screen, maze_pen = track.draw()

            for i, car in enumerate(cars):
                car.replace_track(track)
                car.reset()
                car.mutate(i/(10*len(cars)))

    prev = time.perf_counter()
    try:
        while True:
            frame_counter += 1
            now = time.perf_counter()
            dt = min(0.05, now - prev)    # clamp für Stabilität
            prev = now

            for car in cars:
                car.update(dt)

            screen.update()               # ein globales Update pro Frame
            time.sleep(0.016)             # ~60 FPS
            if frame_counter % 3000 == 0:
                next_generation()
    except (turtle.Terminator, KeyboardInterrupt):
        pass
