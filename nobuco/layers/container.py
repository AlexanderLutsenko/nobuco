from nobuco.commons import TF_TENSOR_CLASSES, ConnectivityStatus
from nobuco.layers.weight import WeightLayer
from nobuco.converters.channel_ordering import TensorPlaceholder, template_insert_recursively
from nobuco.util import collect_recursively


class TransientContainer:
    def __init__(self, op_descr_list, input_names, output_names, outputs_template, constants_dict=None, disconnected_tensors_descr_list=None):
        self.op_descr_list = op_descr_list
        self.input_names = input_names
        self.output_names = output_names
        self.outputs_template = outputs_template
        self.constants_dict = {} if constants_dict is None else constants_dict
        self.disconnected_tensors_descr_list = [] if disconnected_tensors_descr_list is None else disconnected_tensors_descr_list

    @classmethod
    def create(cls, input_names, output_names, outputs_template, disconnected_tensors_keras, children_converted_nodes, constants_to_variables: bool):
        children_descr_list = [(node.input_names, node.output_names, node.keras_op, node.pytorch_node.make_inputs_template()) for node in children_converted_nodes]
        if constants_to_variables:
            const_input_name = input_names[0]
            disconnected_tensors_descr_list = [([const_input_name], [output_name], WeightLayer.create(t, trainable=True), ([TensorPlaceholder(0)], {})) for output_name, t in disconnected_tensors_keras.items()]
            return TransientContainer(children_descr_list, input_names, output_names, outputs_template, constants_dict={}, disconnected_tensors_descr_list=disconnected_tensors_descr_list)
        else:
            return TransientContainer(children_descr_list, input_names, output_names, outputs_template, constants_dict=disconnected_tensors_keras, disconnected_tensors_descr_list=[])

    def _traverse_graph(self, start_names, op_descr_list, reverse_graph):
        traversed_nodes = set(start_names)
        used_nodes = set()
        num_traversed_prev = -1
        while len(traversed_nodes) > num_traversed_prev:
            num_traversed_prev = len(traversed_nodes)

            for input_names, output_names, _, _ in op_descr_list:
                if reverse_graph:
                    input_names, output_names = output_names, input_names

                relevant_inputs = set(input_names).intersection(traversed_nodes)
                if len(relevant_inputs) > 0:
                    # Handle in-place ops
                    relevant_inputs = relevant_inputs.difference(output_names)
                    used_nodes.update(relevant_inputs)
                    traversed_nodes.update(output_names)

        terminal_nodes = traversed_nodes.difference(used_nodes)
        return traversed_nodes, terminal_nodes

    def get_connectivity_status(self) -> ConnectivityStatus:
        traversed_nodes_forward, graph_outputs = self._traverse_graph(self.input_names, self.op_descr_list, reverse_graph=False)
        traversed_nodes_backward, graph_inputs = self._traverse_graph(self.output_names, self.op_descr_list, reverse_graph=True)

        unused_inputs = set(self.input_names).difference(traversed_nodes_backward)
        unreached_outputs = set(self.output_names).difference(traversed_nodes_forward)
        unused_nodes = graph_outputs.difference(set(self.output_names))
        unprovided_inputs = graph_inputs.difference(set(self.input_names))

        return ConnectivityStatus(unused_inputs, unreached_outputs, unused_nodes, unprovided_inputs)

    def __call__(self, *args, training=False, **kwargs):
        inputs = collect_recursively((args, kwargs), TF_TENSOR_CLASSES)

        node_dict = self.constants_dict.copy()

        for input, name in zip(inputs, self.input_names):
            node_dict[name] = input

        for input_names, output_names, op, (args_template, kwargs_template) in (self.disconnected_tensors_descr_list + self.op_descr_list):
            input_tensors = [node_dict[name] for name in input_names]
            args, kwargs = template_insert_recursively((args_template, kwargs_template), input_tensors)
            outputs = op(*args, **kwargs)
            output_tensors = collect_recursively(outputs, TF_TENSOR_CLASSES)
            assert len(output_names) == len(output_tensors)

            for name, output in zip(output_names, output_tensors):
                node_dict[name] = output

        output_tensors = [node_dict[name] for name in self.output_names]
        outputs = template_insert_recursively(self.outputs_template, output_tensors)
        return outputs