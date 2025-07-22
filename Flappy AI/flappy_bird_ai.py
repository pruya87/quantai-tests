import pygame
import neat
import time
import os 
import random
#from lib import *
import pickle
import math

pygame.font.init()

WIN_WIDTH = 550
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
BIRD_FRAMES = [0,1,2,1]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 50)
STAT_FONT_L = pygame.font.SysFont("comicsans", 20)

generation_counter = 0

def random_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) #Avoid dark colors

def normalize(value, min_val, max_val):
    """ Normalize a value between 0 and 1 based on given min and max. """
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

class NEATViewer:
    def __init__(self, genome=None, screen_width=WIN_WIDTH, screen_height=WIN_HEIGHT, output_nodes=None):
        self.genome = genome
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.node_radius = 10
        self.layer_spacing = 10
        self.node_spacing = 10
        #self.step_mode = step_mode  # Start in continuous mode
        #self.wait_for_step = wait_for_step
        self.neuron_pulse = {}  # Store neuron glow effect
        self.activations = {}  # Stores activation values for neurons
        self.activation_decay = 0.95  # Decay factor for glow effect

    def normalize_weight(self, w, min_w, max_w):
        return 1 + 4 * ((abs(w) - min_w) / (max_w - min_w + 1e-5))  # Avoid div by zero

    def draw(self, win, inputs, outputs, genome=None, y=None, output_nodes=None):

        self.output_layer = output_nodes
        self.genome = genome

        self.max_weight = max(abs(conn.weight) for conn in self.genome.connections.values())
        self.min_weight = min(abs(conn.weight) for conn in self.genome.connections.values())

        if self.genome:
            positions = {}
            values = {}
            self.X_GAP = 25
            self.Y_GAP = 45
            self.bird_x = 230 - 150
            self.bird_y = y - 100
            
            neurons = list(set([cg.key[0] for cg in self.genome.connections.values()] + [cg.key[1] for cg in self.genome.connections.values()]))

            self.input_layer = [n for n in neurons if n < 0]
            self.hidden_layer = [n for n in neurons if n >= 0 and n not in output_nodes]
            self.output_layer = [n for n in neurons if n >= 0 and n in output_nodes]

            self.input_neurons = len(self.input_layer)
            self.hidden_neurons = len(self.hidden_layer)
            self.output_neurons = len(self.output_layer)

            self.input_size = self.input_neurons * self.Y_GAP
            self.hidden_size = self.hidden_neurons * self.Y_GAP
            self.output_size = self.output_neurons * self.Y_GAP

            h_center = self.input_size / 2

            # Positioning input nodes
            for i, node_key in enumerate(self.input_layer):
                positions[node_key] = (self.bird_x + self.X_GAP, self.bird_y + i * self.Y_GAP + 100)
                values[node_key] = inputs[i]

            # Positioning hidden nodes
            for i, node_key in enumerate(self.hidden_layer):
                positions[node_key] = (self.bird_x + self.X_GAP * 3,  self.bird_y + (h_center - (self.output_size/2) + i * self.Y_GAP + 100))

            # Positioning output nodes
            for i, node_key in enumerate(self.output_layer):
                positions[node_key] = (self.bird_x + self.X_GAP * 5,  self.bird_y + (h_center - (self.output_size/2) + i * self.Y_GAP + 100))
           
            #self.update_activations()

            # Draw connections
            for conn in self.genome.connections.values():
                start_pos = positions[conn.key[0]]
                end_pos = positions[conn.key[1]]
                weight = conn.weight
                # Thickness based on weight
                thickness = int(self.normalize_weight(weight, self.min_weight, self.max_weight))
                color_intensity = min(255, max(0, int(abs(weight) * 255)))
                color = (0, 255, 0) if weight > 0 else (255, 0, 0)
                pygame.draw.line(win, color, start_pos, end_pos, thickness)

                # Render weight text at midpoint
                mid_pos = ((start_pos[0] + end_pos[0]) // 2 - 25, (start_pos[1] + end_pos[1]) // 2)
                weight_surface = STAT_FONT_L.render(f"{weight:.2f}", True, (255, 255, 255))
                win.blit(weight_surface, mid_pos)

        # Update neuron glow effect
            for node in neurons:
                if node < 0:
                    #activation = self.activations.get(node, 0)
                    self.neuron_pulse[node] = abs(max(self.neuron_pulse.get(node, 0) * self.activation_decay, values[node]))
                else:
                    self.neuron_pulse[node] = abs(max(self.neuron_pulse.get(node, 0) * self.activation_decay, outputs[0]))

            
            # Draw nodes
            min_act = min(self.activations.values(), default=0)
            max_act = max(self.activations.values(), default=1)
            for node_key, position in positions.items():
                # activation = activation = normalize(self.activations.get(node_key, 0), min_act, max_act)
                activation = normalize(outputs[0], min_act, max_act)
                glow = max(0, min(255, int(255 * self.neuron_pulse.get(node_key, 0))))
                #color = (int(activation * 255), int(activation * 255), 255)
                color = (glow, glow, 255) if node_key in self.output_layer else (glow, 255, glow)
#                pygame.draw.circle(win, (255, 255, 255), position, self.node_radius)
                pygame.draw.circle(win, color, position, self.node_radius)
                pygame.draw.circle(win, (0, 255, 0), position, self.node_radius, 2)

                if node_key >= 0:  # Only for non-input neurons
                    node = self.genome.nodes[node_key]
                    activation_func = node.activation  # Get activation function name
                    bias = node.bias

                    # Render and draw text
                    text_surface = STAT_FONT_L.render(f"{activation_func}, Bias: {bias:.2f}", True, (255, 255, 255))
                    win.blit(text_surface, (position[0] - 20, position[1] - 80))
                    # Display neuron id & activation value
                    font = pygame.font.Font(None, 24)
                    text = STAT_FONT_L.render(f"{node_key} : {activation:.2f}", True, (255, 255, 255))
                    win.blit(text, (position[0] - 10, position[1] - 30))
                else:
                    input = STAT_FONT_L.render(f"{values[node_key]: .2f}", True, (255, 255, 255))
                    win.blit(input, (position[0] - 50, position[1] - 30))
        else:
            print("No genome found to visualize!")
        
class NEATAnalyzer:
	def __init__(self, i=None, net=None, genome=None, output_nodes=None):
		self.net = net
		self.genome = genome
		self.output_nodes = output_nodes
		self.i = i

	def print_gene_fitness(self, i=None, genome=None):
		self.genome = genome
		print(f"Top {i+1} - Fitness: {self.genome.fitness}")	

	def print_node_evals(self, net=None):
		self.net = net
		"""Prints the node evaluations from the network."""
		for node_id, activation, aggregation, bias, response, links in self.net.node_evals:
			print(f"Node {node_id}: Activation={activation.__name__}, Bias={bias}, Response={response}")
			for link in links:
				print(f"  Connected to {link[0]} with weight {link[1]}")

	def print_nodes_values(self, genome=None):
		self.genome = genome
		"""Prints all node values from the genome."""
		for node in self.genome.nodes.values():
			print(node)

	def print_connections_values(self, genome=None):
		self.genome = genome
		"""Prints all connection values from the genome."""
		for connection in self.genome.connections.values():
			print(connection)

	def print_nodes_items(self, genome=None, output_nodes=None):
		self.genome = genome
		self.output_nodes = output_nodes
		"""Prints node items and determines if they are hidden or output nodes."""
		for node_id, node in self.genome.nodes.items():
			if node_id in self.output_nodes:
				print(f"Node {node_id} {node} is an OUTPUT node.")
			else:
				print(f"Node {node_id} {node} is a HIDDEN node.")

class Pipe:
	GAP = 200
	VEL = 5

	def __init__(self, x):
		self.x = x
		self.height = 0
		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
		self.PIPE_BOTTOM = PIPE_IMG

		self.passed = False
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):
		self.x -= self.VEL

	def draw(self, win):
		win.blit(self.PIPE_TOP, (self.x, self.top))
		win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

	def collide(self, bird):
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		top_offset = (self.x - bird.x, self.top -round(bird.y))
		bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

		b_point = bird_mask.overlap(bottom_mask, bottom_offset)
		t_point = bird_mask.overlap(top_mask, top_offset)

		if t_point or b_point:
			return True
		else:
			return False
		
class Base:
	VEL = 5
	WIDTH = BASE_IMG.get_width()
	IMG = BASE_IMG

	def __init__(self, y):
		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH
	
	def move(self):
		self.x1 -= self.VEL
		self.x2 -= self.VEL

		if self.x1 + self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH
		if self.x2 + self.WIDTH < 0:
			self.x2 = self.x1 + self.WIDTH

	def draw(self, win):
		win.blit(self.IMG, (self.x1, self.y))
		win.blit(self.IMG, (self.x2, self.y))

class Bird:
	IMGS = BIRD_IMGS
	MAX_ROTATION = 25
	ROT_VEL = 20
	ANIMATION_TIME = 5
	inputs = ()
	output = 0

	def __init__(self, x, y, color=(255, 255, 255)):
		self.x = x
		self.y = y
		self.tilt = 0
		self.tick_count = 0
		self.vel = 0
		self.height = self.y
		self.img_count = 0
		self.img = self.IMGS[0]
		self.color = color

	def jump(self):
		self.vel = -10.5
		#When we jump we reset the movement timee
		self.tick_count = 0
		self.height = self.y

	def move(self):
		self.tick_count += 1

		d = self.vel * self.tick_count + 1.5 * self.tick_count**2
		
		# 16 is the terminal velocity. We can't move downwards faster
		if d >= 16:
			d = 16
		elif d < 0:
			d -= 2

		self.y = self.y + d

		if d < 0 or self.y < self.height + 50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION
		elif self.tilt > -90:
			self.tilt -= self.ROT_VEL

	def draw(self, win):
		self.img_count += 1
		self.img = self.IMGS[BIRD_FRAMES[self.img_count%3]]
		#self.img = self.tint_image(self.IMGS[BIRD_FRAMES[self.img_count%3]], self.color)
		
		if self.tilt <= -80:
			self.img = self.tint_image(self.IMGS[1], self.color)
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME*2

		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
		win.blit(rotated_image , new_rect.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.img)

	def tint_image(self, image, color, intensity=128):
		""" Apply a subtle color tint to an image without losing details. """
		tinted_image = image.copy()


		# Create a colored surface with some transparency
		color_surface = pygame.Surface(image.get_size(), pygame.SRCALPHA)
		color_surface.fill(color + (intensity,))  # Add transparency (intensity controls effect)

		# Blend the original image with the colored overlay
		tinted_image.blit(color_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
		return tinted_image

def draw_nn(win, net, x, y):
	"""Draws a simple visualization of a neural network."""
	radius = 10
	input_nodes = [(x, y + i * 30) for i in range(3)]
	output_nodes = [(x + 150, y + 15)]

	# Draw input and output nodes
	for node in input_nodes + output_nodes:
		pygame.draw.circle(win, (255, 255, 255), node, radius)

	# Draw connections
	for i, input_node in enumerate(input_nodes):
		connection = net.node_evals[i][5]  # Get connection tuple
		weight = connection[1]  # Weight is the second element in the tuple
		color = (0, 255, 0) if weight > 0 else (255, 0, 0)
		pygame.draw.line(win, color, input_node, output_nodes[0], 2)

def draw_window(win, birds, pipes, base, score, alive, nets, alphas, output_nodes):
	na = NEATAnalyzer()
	nv = NEATViewer()
	win.blit(BG_IMG, (0, 0))  # Clear screen
	for pipe in pipes:
		pipe.draw(win)
    
	base.draw(win)
	for bird in birds:
		bird.draw(win)

	for i, (genome, bird, net) in enumerate(alphas):
		#print(f"Alpha: {i}, X: {bird.x}, Y: {bird.y}")
		#na.print_gene_fitness(i, genome)
		#na.print_node_evals(net)
		#na.print_nodes_values(genome)
		#na.print_connections_values(genome, output_nodes)
		#nv.update_activations(bird.inputs, bird.output)
		nv.draw(win, bird.inputs, bird.output, genome, bird.y, output_nodes)

	text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
	text = STAT_FONT.render("Survivors: " + str(alive), 1, (225,225,225))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 40))

	# Draw neural networks of alive birds
	#for i, bird in enumerate(birds[:3]):  # Limit to 3 birds to avoid clutter
	#    draw_nn(win, nets[i], 500, 100 + i * 100)

	pygame.display.update()

def get_top_agents(ge, birds, nets, top_n=3):
    """Returns the top N birds based on fitness in real-time."""
    sorted_agents = sorted(zip(ge, birds, nets), key=lambda x: x[0].fitness, reverse=True)
    return sorted_agents[:top_n]  # Return top N agents

def main(genomes, config):
	global generation_counter
	global step_mode 
	global wait_for_step 
	step_mode = False
	wait_for_step = False
	generation_counter += 1
	print(f"Starting generation... ", {generation_counter})
	nets = []
	ge = []
	birds = []
	output_nodes = set(config.genome_config.output_keys)

	for _, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		birds.append(Bird(230, 350))
		g.fitness = 0
		ge.append(g)

	base = Base(730)
	pipes = [Pipe(700)]
	score = 0

	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	clock = pygame.time.Clock()

	run = True
	while run:
		clock.tick(30)
		add_pipe = False
		rem = []
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					bird.jump()
				if event.key == pygame.K_s:  # Switch to step-by-step mode
					step_mode = True
					wait_for_step = True
				elif event.key == pygame.K_c:  # Switch to continuous mode
					step_mode = False
					wait_for_step = False
				elif event.key == pygame.K_n and step_mode:  # Advance one step
					wait_for_step = False

		pipe_ind = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind=1
		else: #If we have no birds left we skip to the next generation
			#run = False
			#break
			return
	
		for x, bird in enumerate(birds):
			bird.move()
			ge[x].fitness += 0.1

			bird.inputs = (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))
			bird.output = nets[x].activate(bird.inputs)
			# If the output neuron 0 is more than 0.5 the bird jumps
			if bird.output[0] > 0.5:
				bird.jump()


		#bird.move()
		base.move()
		for pipe in pipes:
			for x, bird in enumerate(birds):
				if pipe.collide(bird):
					ge[x].fitness -= 1
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)
				if not pipe.passed and pipe.x + pipe.PIPE_TOP.get_width() < bird.x:
					pipe.passed = True
					add_pipe = True
				if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)
			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)

			pipe.move()

		if add_pipe:
			score += 1 
			for g in ge:
				g.fitness += 5
			pipes.append(Pipe(700))
		for r in rem:
			pipes.remove(r)

		#for ix in range(6):
		#	print(f"Index {ix} is:")
		#	print(nets[0].node_evals[0][ix])
		
		alphas = get_top_agents(ge, birds, nets)

		if not step_mode or not wait_for_step:
			# Update and draw the game only if not waiting for step confirmation
			draw_window(win, birds, pipes, base, score, len(birds), nets, alphas, output_nodes)
			wait_for_step = True  # Reset step wait in step mode


#main()

def run(config_path):
	print("Running NEAT...")
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
	p = neat.Population(config)
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	#We run the main function(game) with the genomes the genetic algorithm passes us for 50 generations
	winner = p.run(main,20)
	#node_names = {-1: 'A', -2: 'B', -3: 'B', 0: 'A XOR B'}
	#visualize.draw_net(config, winner, True, node_names=node_names)
	print("Finished NEAT execution...")

if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
	run(config_path)



"""
visualization class

>>> for i in ge[0].connections.values():
...     print(i)
...
DefaultConnectionGene(key=(-1, 0), weight=0.18496911576516364, enabled=True)
DefaultConnectionGene(key=(-2, 0), weight=0.2572587960689565, enabled=True)
DefaultConnectionGene(key=(-3, 0), weight=-1.246990018031149, enabled=True)


>>> for i in ge[0].nodes.values():
...     print(i)
...
DefaultNodeGene(key=0, bias=1.051017942485378, response=1.0, activation=tanh, aggregation=sum)

for node_id, activation, aggregation, bias, response, links in nets[0].node_evals:
	print(f"Node {node_id}: Activation={activation.__name__}, Bias={bias}, Response={response}")
	for link in links:
		print(f"  Connected to {link[0]} with weight {link[1]}")

Node 0: Activation=tanh_activation, Bias=-0.6431862434862348, Response=1.0
  Connected to -1 with weight -0.17340422874601488
  Connected to -2 with weight 1.595868168437206
  Connected to -3 with weight -0.7638686185205962

for node_id, node in ge[0].nodes.items():
    if node_key in config:
        print(f"Node {node_key} {node} is an OUTPUT node.")
    else:
        print(f"Node {node_key} {node} is a HIDDEN node.")

We save the best genome in a file 
	with open("nn.txt","wb") as f:
		pickle.dump(winner, f) 
	f.close()

"""


