# To run this file, type in
# manim -p neuralnetwork.py NeuralNetworkExample
# into your command line in the directory where this file is stored
from manim import *

import itertools as it

# These config settings allows the video to be vertical, in a 9:16 aspect ratio
config.frame_width = 9 
config.frame_height = 16

config.pixel_width = 1080 
config.pixel_height = 1920

# The main neural network class, where the configuration can be changed
class NeuralNet(VGroup):

    # Change these settings to modify the appearance of the neural network
    CONFIG = {
        "neuron_radius": 0.4,
        "neuron_to_neuron_buff": MED_LARGE_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "output_neuron_color": WHITE,
        "input_neuron_color": WHITE,
        "hidden_layer_neuron_color": WHITE,
        "neuron_stroke_width": 4,
        "neuron_fill_color": "#84ffbc", # Can be hex code
        "edge_color": WHITE,
        "edge_stroke_width": 1.6,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 12,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,    
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 0.9
    }

    def __init__(self, neural_network, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.layer_sizes = neural_network
        self.neuron_radius = self.CONFIG["neuron_radius"]
        self.neuron_to_neuron_buff = self.CONFIG["neuron_to_neuron_buff"]
        self.layer_to_layer_buff = self.CONFIG["layer_to_layer_buff"]
        self.output_neuron_color = self.CONFIG["output_neuron_color"]
        self.input_neuron_color = self.CONFIG["input_neuron_color"]
        self.hidden_layer_neuron_color = self.CONFIG["hidden_layer_neuron_color"]
        self.neuron_stroke_width = self.CONFIG["neuron_stroke_width"]
        self.neuron_fill_color = self.CONFIG["neuron_fill_color"]
        self.edge_color = self.CONFIG["edge_color"]
        self.edge_stroke_width = self.CONFIG["edge_stroke_width"]
        self.edge_propogation_color = self.CONFIG["edge_propogation_color"]
        self.edge_propogation_time = self.CONFIG["edge_propogation_time"]
        self.max_shown_neurons = self.CONFIG["max_shown_neurons"]
        self.brace_for_large_layers = self.CONFIG["brace_for_large_layers"]
        self.average_shown_activation_of_large_layer = self.CONFIG["average_shown_activation_of_large_layer"]
        self.include_output_labels = self.CONFIG["include_output_labels"]
        self.arrow = self.CONFIG["arrow"]
        self.arrow_tip_size = self.CONFIG["arrow_tip_size"]
        self.left_size = self.CONFIG["left_size"]
        self.neuron_fill_opacity = self.CONFIG["neuron_fill_opacity"]
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        self.add(layers)
        if self.include_output_labels:
            self.label_outputs_text()

    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color

    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = min(size, self.max_shown_neurons)
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=self.neuron_fill_color,
                fill_opacity=self.neuron_fill_opacity,
            )
            for _ in range(n_neurons)
        ])
        neurons.arrange(DOWN, buff=self.neuron_to_neuron_buff)
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = MathTex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(dots, UP, MED_SMALL_BUFF)
            VGroup(*neurons[len(neurons) // 2:]).next_to(dots, DOWN, MED_SMALL_BUFF)
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add(self.edge_groups)

    def get_edge(self, neuron1, neuron2):
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.edge_color,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )
    
    def label_inputs(self, l):
        self.input_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = MathTex(f"{l}_{{{n + 1}}}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.input_labels.add(label)
        self.add(self.input_labels)

    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(f"{l}_{{{n + 1}}}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(outputs[n])
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width() / 2) * RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_hidden_layers(self, l):
        self.hidden_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = MathTex(f"{l}_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.hidden_labels.add(label)
        self.add(self.hidden_labels)


class NeuralNetworkExample(Scene):
    def construct(self):
        # Create a Neural Network Object with 4 layers, with 4 nodes, 5 nodes, 5 nodes, then 1 node
        network = NeuralNet([4, 5, 5, 1]) 

        # Label the inputs with the letter x
        network.label_inputs('x'); 

        # Label the outputs with yhat
        network.label_outputs('\hat{y}'); 

        # Create the network on the screen
        self.play(Write(network), run_time = 3) 

        # Wait 2 seconds
        self.wait(2) 

        # Shift the network to the left
        self.play(network.animate.shift(LEFT * 3), run_time=1.5)

        # Create a yhat label for the prediction
        y_hat_label = MathTex('\hat{y}', font_size=36).next_to(network.layers[-1], RIGHT, buff=MED_LARGE_BUFF)

        # Draw an arrow from the output neuron to a prediction
        arrow = Arrow(network.layers[-1].neurons.get_center() + RIGHT * 0.2, y_hat_label.get_center(), buff=0.1, stroke_width=2, color=WHITE)

        # Animate it
        self.play(GrowArrow(arrow), Write(y_hat_label))

        # Wait 2 seconds
        self.wait(2)


# This is a template, so feel free to change whatever parameters to achieve the effect you want!