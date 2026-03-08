import turtle
import random
import copy

class Neuron:
    def __init__(self, weights):
        self.weights = weights
    
    def activate(self, inputs):
        return sum([input * weight for input, weight in zip(inputs, self.weights)])
            
class Network:,


    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.neurons = []
        for i in range(hidden_size):
            weights = []
            for i in range(hidden_size):
                weights.append(random.uniform(-1, 1))
            self.neurons.append(Neuron(weights))
        for i in range(output_size):
            weights = []
            for i in range(hidden_size):
                weights.append(random.uniform(-1, 1))
            self.neurons.append(Neuron(weights))
    
    def forward(self, inputs):
        hidden = [neuron.activate(inputs) for neuron in self.neurons[:self.hidden_size]]
        output = [neuron.activate(hidden) for neuron in self.neurons[self.hidden_size:]]
        return output

    def mutate(self, factor):
        for neuron in self.neurons:
            for i in range(len(neuron.weights)):
                if random.random() < 1:
                    neuron.weights[i] += random.uniform(-1*factor, 1*factor)

class Base:
    def __init__(self, screen):
        self.screen = screen
        self.turtle = turtle.Turtle()
        self.turtle.penup()
    
    def move(self, direction):
        dir_dict = {"up": 90, "down": 270, "left": 180, "right": 0}
        self.turtle.setheading(dir_dict[direction])
        self.turtle.forward(10)
    
    def position(self):
        return self.turtle.xcor(), self.turtle.ycor()

class Hunter(Base):
    def __init__(self, screen):
        super().__init__(screen)
        self.turtle.color("red")
        self.score = 0
        self.age = 0
        self.brain = Network(2, 4, 4)
        self.prev_dist = 0
    
    def hunt(self, dot):
        dx = (dot.position()[0] - self.position()[0])/200
        dy = (dot.position()[1] - self.position()[1])/200
        inputs = [dx, dy]
        output = self.brain.forward(inputs)
        idx = output.index(max(output))
        if idx == 0:
            self.move("right")
        elif idx == 1:
            self.move("left")
        elif idx == 2:
            self.move("up")
        elif idx == 3:
            self.move("down")
        distance = ((dx*200)**2 + (dy*200)**2)**0.5
        if distance < 10:
            self.score += 4
        elif distance < 20:
            self.score += 2
        elif distance < 30:
            self.score += 1
        elif distance < 40:
            self.score += 0.5
        elif distance < 50:
            self.score += 0.1
        self.score += (self.prev_dist - distance)
        self.prev_dist = distance

    def mutate(self):
        self.age += 1
        max = 4*90
        factor = (0.99**self.age)*(max - self.score)/max
        self.brain.mutate(factor)

    def clone(self):
        child = Hunter(self.screen)          
        child.brain = copy.deepcopy(self.brain)  
        child.score = 0
        child.age = 0
        child.turtle.goto(0, 0)
        return child

class Dot(Base):
    def __init__(self, screen):
            super().__init__(screen)
            self.x = 0 #random.randint(-200, 200)
            self.y = 200#random.randint(-200, 200)
            self.turtle.shape("circle")
            self.turtle.color("blue")
            self.turtle.goto(self.x, self.y)
        
    def random_start(self):
        heading = random.choice([90, 270, 180, 0])
        self.turtle.setheading(heading)
        if heading == 0:
            self.x, self.y = 0, 200
        elif heading == 90:
            self.x, self.y = -200, 0
        elif heading == 180:
            self.x, self.y = 0, -200
        elif heading == 270:
            self.x, self.y = 200, 0
        self.turtle.goto(self.x, self.y)
        
        
    def move_random(self):
        if self.turtle.xcor() > 200:
            self.move("left")
            return
        elif self.turtle.xcor() < -200:
            self.move("right")
            return
        elif self.turtle.ycor() > 200:
            self.move("down")
            return
        elif self.turtle.ycor() < -200:
            self.move("up")
            return
        self.move(random.choice(["up", "down", "left", "right"]))

    def move_circle(self):
        self.turtle.forward(4)
        self.turtle.right(1)



screen = turtle.Screen()
screen.setup(width=800, height=600)
screen.title("Seek the Dot")
screen.bgcolor("white")
screen.tracer(0)

hunter = Hunter(screen)
dot = Dot(screen)
counter = 0

hunters = []
for _ in range(20):
    hunters.append(Hunter(screen))

def next_generation():
    global hunters
    global dot
    hunters.sort(key=lambda hunter: hunter.score, reverse=True)
    losers = hunters[10:]
    for loser in losers:
        loser.turtle.hideturtle()
        del(loser)
    hunters = hunters[:10]
    for i in range(10):
        hunters.append(hunters[i].clone())
    for i in range(5):
        hunters.append(hunters[i].clone())
    for hunter in hunters:
        hunter.score = 0
        hunter.turtle.goto(0, 0)
        hunter.mutate()
    dot.random_start()


def tick():
    global counter
    dot.move_circle()
    for hunter in hunters:
        hunter.hunt(dot)
    if counter % 90 == 0:
        next_generation()
    counter += 1
    # print(dot.position())
    screen.update()
    screen.ontimer(tick, 16) 

screen.listen()
tick()
screen.mainloop()