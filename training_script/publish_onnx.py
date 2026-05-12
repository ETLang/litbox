import onnx
import json
import numpy as np
import argparse

def publish_onnx(onnx_path, weights_path, json_path):
    """
    Processes an ONNX file and publishes it into a binary weights blob and a
    JSON file describing the model's architecture for use in Unity.
    """
    def json_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        model = onnx.load(onnx_path)
        graph = model.graph
    except Exception as e:
        print(f"Error loading ONNX file: {e}")
        return

    # --- 1. Extract all initializers (weights, biases, etc.) ---
    initializers = {init.name: init for init in graph.initializer}

    # --- 2. Build the weights blob and operation list ---
    weights_blob = bytearray()
    ops_list = []
    
    # The nodes in the ONNX graph are already topologically sorted.
    for node in graph.node:
        op_type = node.op_type

        # Filter for only relevant operations
        supported_ops = ['Add', 'Concat', 'Conv', 'DepthToSpace', 'MaxPool', 'Pad', 'Relu']
        if op_type not in supported_ops:
            print(f"Info: Skipping unsupported ONNX operation type '{op_type}'.")
            continue

        final_op_info = {
            'type': node.op_type,
            'name': node.name,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'params': {}
        }
        
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        matched = False

        if op_type == 'Pad':
            pads = []
            # Pad amounts can be a constant input (opset >= 11)
            if len(node.input) > 1 and node.input[1] in initializers:
                tensor = initializers[node.input[1]]
                pads = onnx.numpy_helper.to_array(tensor).tolist()
            mode = attrs.get('mode', 'constant')
            if isinstance(mode, bytes):
                mode = mode.decode('utf-8')
            
            if mode == 'edge' and pads == [0, 0, 1, 1, 0, 0, 1, 1]:
                final_op_info['type'] = 'ClampPad'
                if len(node.input) > 1 and node.input[1] in initializers:
                    final_op_info['inputs'].remove(node.input[1])
                matched = True
            else:
                final_op_info['params']['pads'] = pads
                final_op_info['params']['mode'] = mode
                if len(node.input) > 1 and node.input[1] in initializers:
                    final_op_info['inputs'].remove(node.input[1])

        elif op_type == 'Relu':
            final_op_info['type'] = 'Relu'
            matched = True

        elif op_type == 'Add':
            final_op_info['type'] = 'Add'
            matched = True
            # Check if one of the inputs is a constant initializer
            for input_name in list(final_op_info['inputs']): # Iterate over a copy
                if input_name in initializers:
                    tensor = initializers[input_name]
                    np_const = onnx.numpy_helper.to_array(tensor)
                    offset = len(weights_blob)
                    weights_blob.extend(np_const.tobytes())
                    size = len(weights_blob) - offset
                    final_op_info['params']['constant'] = {
                        'offset': offset,
                        'size': size,
                        'shape': list(np_const.shape),
                        'dtype': str(np_const.dtype)
                    }
                    final_op_info['inputs'].remove(input_name)

        elif op_type == 'MaxPool':
            kernel_shape = attrs.get('kernel_shape')
            strides = attrs.get('strides', [1] * len(kernel_shape))
            pads = attrs.get('pads', [0] * 2 * len(kernel_shape))
            
            if kernel_shape == [2, 2] and strides == [2, 2] and pads == [0, 0, 0, 0]:
                final_op_info['type'] = 'BasicMaxPool'
                matched = True
            else:
                final_op_info['params']['kernel_shape'] = kernel_shape
                final_op_info['params']['strides'] = strides
                final_op_info['params']['pads'] = pads

        elif op_type == 'Concat':
            axis = attrs.get('axis')
            if axis == 1:
                final_op_info['type'] = 'ConcatChannels'
                matched = True
            else:
                final_op_info['params']['axis'] = axis

        elif op_type == 'DepthToSpace':
            blocksize = attrs.get('blocksize')
            if blocksize == 2:
                final_op_info['type'] = 'BasicPixelShuffle'
                matched = True
            else:
                final_op_info['params']['blocksize'] = blocksize

        elif op_type == 'Conv':
            weight_name = node.input[1]
            if weight_name in initializers:
                weight_tensor = initializers[weight_name]
                np_weights = onnx.numpy_helper.to_array(weight_tensor)
                kernel_dims = len(np_weights.shape) - 2
                dilations = attrs.get('dilations', [1] * kernel_dims)
                group = attrs.get('group', 1)
                pads = attrs.get('pads', [0] * 2 * kernel_dims)
                strides = attrs.get('strides', [1] * kernel_dims)

                # If pads are [1, 1, 1, 1], inject a ClampPad before this Conv
                if pads == [1, 1, 1, 1]:
                    clamp_pad_op = {
                        'type': 'ClampPad',
                        'name': node.name + '_pre_clamp_pad',
                        'inputs': [node.input[0]],
                        'outputs': [node.name + '_clamped'],
                        'params': {}
                    }
                    ops_list.append(clamp_pad_op)
                    
                    # Update current Conv to use the output of the new ClampPad
                    final_op_info['inputs'][0] = clamp_pad_op['outputs'][0]
                    # Reset pads to zero as they are now handled by ClampPad
                    pads = [0, 0, 0, 0]

                # --- Perform matching ---
                if group == 1 and pads == [0, 0, 0, 0]:
                    if dilations == [1, 1] and strides == [1, 1]:
                        final_op_info['type'] = 'BasicConv'
                        matched = True
                    else:
                        final_op_info['type'] = 'ComplexConv'
                        final_op_info['params']['strides'] = strides
                        final_op_info['params']['dilations'] = dilations
                        matched = True
                
                # --- Extract weights/bias ---
                offset = len(weights_blob)
                weights_blob.extend(np_weights.tobytes())
                size = len(weights_blob) - offset
                final_op_info['params']['weights'] = {
                    'offset': offset, 'size': size, 'shape': list(np_weights.shape), 'dtype': str(np_weights.dtype)
                }
                final_op_info['inputs'].remove(weight_name)

                if len(node.input) > 2 and node.input[2] in initializers:
                    bias_name = node.input[2]
                    tensor = initializers[bias_name]
                    np_bias = onnx.numpy_helper.to_array(tensor)
                    offset = len(weights_blob)
                    weights_blob.extend(np_bias.tobytes())
                    size = len(weights_blob) - offset
                    final_op_info['params']['bias'] = {
                        'offset': offset, 'size': size, 'shape': list(np_bias.shape), 'dtype': str(np_bias.dtype)
                    }
                    final_op_info['inputs'].remove(bias_name)
                
                if not matched:
                    final_op_info['params']['dilations'] = dilations
                    final_op_info['params']['group'] = group
                    final_op_info['params']['pads'] = pads
                    final_op_info['params']['strides'] = strides

        # --- Finalize UNMATCHED cases ---
        if not matched:
            print(f"Warning: UNMATCHED operation found: {node.name} (type: {op_type})")
            final_op_info['type'] = 'UNMATCHED'
            final_op_info['original_type'] = op_type
        
        ops_list.append(final_op_info)

    # --- 3. Print un-consolidated graph for validation ---
    print("\nUn-consolidated Graph Representation:")
    print("-" * 40)
    
    output_to_idx_unconsolidated = {}
    for i, op in enumerate(ops_list):
        for out_name in op['outputs']:
            output_to_idx_unconsolidated[out_name] = i + 1

    for i, op in enumerate(ops_list):
        input_indices = [str(output_to_idx_unconsolidated.get(inp, 0)) for inp in op['inputs']]
        inputs_str = ", ".join(input_indices)
        print(f"{i+1}. [{op['type']}] (Inputs: {inputs_str})")
    print("-" * 40)

    # --- 4. Consolidate operations into ResidualConv and BasicMaxPool ---
    output_to_op = {}
    tensor_consumers = {}
    for op in ops_list:
        for out in op['outputs']:
            output_to_op[out] = op
        for inp in op['inputs']:
            if inp not in tensor_consumers:
                tensor_consumers[inp] = []
            tensor_consumers[inp].append(op)

    consolidated_ops = []
    consumed_op_names = set()
    residual_group = []
    residual_state = 0

    def consume_residual_group():
        nonlocal residual_group, residual_state
        if not residual_group:
            return
        
        residual_state = 0
        concat_op = None
        for op in residual_group:
            if 'ConcatChannels' == op['type']:
                concat_op = op
                break

        clamp_op = None
        for op in residual_group:
            if 'ClampPad' == op['type']:
                clamp_op = op
                break

        conv_op = None
        for op in residual_group:
            if 'BasicConv' == op['type']:
                conv_op = op
                break

        add_op = None
        for op in residual_group:
            if 'Add' == op['type']:
                add_op = op
                break

        relu_op = None
        for op in residual_group:
            if 'Relu' == op['type']:
                relu_op = op
                break
        
        shuffle_op = None
        for op in residual_group:
            if 'BasicPixelShuffle' == op['type']:
                shuffle_op = op
                break

        if conv_op == None:
            # This group does not contain a Conv, so just add the ops as-is.
            for op in residual_group:
                if op['name'] not in consumed_op_names:
                    consolidated_ops.append(op)
                    consumed_op_names.add(op['name'])
            return
        
        # Verify that the inputs/outputs of the ops in residual_group align with the sequence of ops in the list.
        # This ensures that the operations are actually connected in a chain before fusing them.
        for j in range(len(residual_group) - 1):
            current_op = residual_group[j]
            next_op = residual_group[j+1]
            if not any(out in next_op['inputs'] for out in current_op['outputs']):
                # The chain is broken, so we cannot safely fuse this as a single sequence.
                for op in residual_group:
                    if op['name'] not in consumed_op_names:
                        consolidated_ops.append(op)
                        consumed_op_names.add(op['name'])
                residual_group = []
                return

        # This group should be fused into a single ResidualConv.
        # The fusion logic below is adapted from the original 'BasicConv' block.
        # It starts from the conv_op and traces backwards and forwards to find fusable neighbors.
        op = conv_op  # Use 'op' to match the original code's variable names

        residual_conv = {
            'type': 'ResidualConv',
            'name': op['name'],
            'inputs': list(op['inputs']),
            'outputs': list(op['outputs']),
            'params': {
                'weights': op['params'].get('weights'),
                'bias': op['params'].get('bias'),
                'strides': op['params'].get('strides'),
                'dilations': op['params'].get('dilations'),
                'relu': False,
                'pixelShuffle2x2': False,
            }
        }

        if clamp_op != None:
            residual_conv['params']['pad'] = 'clamp'
            residual_conv['inputs'][0] = clamp_op['inputs'][0]

        if concat_op != None:
            residual_conv['params']['concat'] = concat_op['inputs'][1]
            residual_conv['inputs'][0] = concat_op['inputs'][0]

        if add_op != None:
            residual_conv['params']['add'] = add_op['inputs'][1]
            residual_conv['outputs'][0] = add_op['outputs'][0]

        if relu_op != None:
            residual_conv['params']['relu'] = True
            residual_conv['outputs'][0] = relu_op['outputs'][0]

        if shuffle_op != None:
            residual_conv['params']['pixelShuffle2x2'] = True
            residual_conv['outputs'][0] = shuffle_op['outputs'][0]
        
        # consumed_op_names.add(op['name'])

        # # --- Trace backwards for pre-operations ---
        # current_input_tensor = op['inputs'][0]

        # if current_input_tensor in output_to_op:
        #     prev_op = output_to_op[current_input_tensor]
        #     consumers = tensor_consumers.get(current_input_tensor, [])
        #     if prev_op['name'] not in consumed_op_names and prev_op['type'] == 'ClampPad' and len(consumers) == 1:
        #         residual_conv['params']['pad'] = 'clamp'
        #         consumed_op_names.add(prev_op['name'])
        #         current_input_tensor = prev_op['inputs'][0]

        # if current_input_tensor in output_to_op:
        #     prev_op = output_to_op[current_input_tensor]
        #     consumers = tensor_consumers.get(current_input_tensor, [])
        #     if prev_op['name'] not in consumed_op_names and prev_op['type'] == 'ConcatChannels' and len(consumers) == 1:
        #         residual_conv['inputs'] = [prev_op['inputs'][0]]
        #         residual_conv['params']['concat'] = prev_op['inputs'][1]
        #         consumed_op_names.add(prev_op['name'])
        #     else:
        #         residual_conv['inputs'] = [current_input_tensor]
        # else:
        #     residual_conv['inputs'] = [current_input_tensor]

        # # --- Trace forwards for post-operations ---
        # current_output_tensor = op['outputs'][0]

        # consumers = tensor_consumers.get(current_output_tensor, [])
        # if len(consumers) == 1:
        #     next_op = consumers[0]
        #     if next_op and next_op['name'] not in consumed_op_names and next_op['type'] == 'Add':
        #         add_inputs = list(next_op['inputs'])
        #         add_inputs.remove(current_output_tensor)
        #         residual_conv['params']['add'] = add_inputs[0]
        #         consumed_op_names.add(next_op['name'])
        #         current_output_tensor = next_op['outputs'][0]

        # consumers = tensor_consumers.get(current_output_tensor, [])
        # if len(consumers) == 1:
        #     next_op = consumers[0]
        #     if next_op and next_op['name'] not in consumed_op_names and next_op['type'] == 'Relu':
        #         residual_conv['params']['relu'] = True
        #         consumed_op_names.add(next_op['name'])
        #         current_output_tensor = next_op['outputs'][0]

        # consumers = tensor_consumers.get(current_output_tensor, [])
        # if len(consumers) == 1:
        #     next_op = consumers[0]
        #     if next_op and next_op['name'] not in consumed_op_names and next_op['type'] == 'BasicPixelShuffle':
        #         residual_conv['params']['pixelShuffle2x2'] = True
        #         consumed_op_names.add(next_op['name'])
        #         current_output_tensor = next_op['outputs'][0]

        # residual_conv['outputs'] = [current_output_tensor]
        consolidated_ops.append(residual_conv)
        
        residual_group = []

    for op in ops_list:
        if op['name'] in consumed_op_names:
            continue

        if op['type'] == 'BasicMaxPool':
            consume_residual_group()
            consolidated_ops.append(op)
            consumed_op_names.add(op['name'])
            continue

        if op['type'] == 'ConcatChannels':
            if residual_state != 0:
                consume_residual_group()
            residual_group.append(op)
            residual_state = 1
            continue

        if op['type'] == 'ClampPad':
            if residual_state >= 2:
                consume_residual_group()
            residual_group.append(op)
            residual_state = 2
            continue

        if op['type'] == 'BasicConv':
            if residual_state >= 3:
                consume_residual_group()
            residual_group.append(op)
            residual_state = 3
            continue

        if op['type'] == 'Add':
            if residual_state >= 4:
                consume_residual_group()
            residual_group.append(op)
            residual_state = 4
            continue

        if op['type'] == 'Relu':
            if residual_state >= 5:
                consume_residual_group()
            residual_group.append(op)
            residual_state = 5
            continue

        if op['type'] == 'BasicPixelShuffle':
            if residual_state >= 6:
                consume_residual_group()
            residual_group.append(op)
            residual_state = 6
            continue

        # if op['type'] == 'BasicConv':
        #     residual_conv = {
        #         'type': 'ResidualConv',
        #         'name': op['name'],
        #         'inputs': list(op['inputs']),
        #         'outputs': list(op['outputs']),
        #         'params': {
        #             'weights': op['params'].get('weights'),
        #             'bias': op['params'].get('bias'),
        #             'strides': op['params'].get('strides'),
        #             'dilations': op['params'].get('dilations'),
        #             'relu': False,
        #             'pixelShuffle2x2': False,
        #         }
        #     }
        #     consumed_op_names.add(op['name'])
            
        #     # --- Trace backwards for pre-operations ---
        #     current_input_tensor = op['inputs'][0]
            
        #     if current_input_tensor in output_to_op:
        #         prev_op = output_to_op[current_input_tensor]
        #         consumers = tensor_consumers.get(current_input_tensor, [])
        #         if prev_op['name'] not in consumed_op_names and prev_op['type'] == 'ClampPad' and len(consumers) == 1:
        #             residual_conv['params']['pad'] = 'clamp'
        #             consumed_op_names.add(prev_op['name'])
        #             current_input_tensor = prev_op['inputs'][0]

        #     if current_input_tensor in output_to_op:
        #         prev_op = output_to_op[current_input_tensor]
        #         consumers = tensor_consumers.get(current_input_tensor, [])
        #         if prev_op['name'] not in consumed_op_names and prev_op['type'] == 'ConcatChannels' and len(consumers) == 1:
        #             residual_conv['inputs'] = [prev_op['inputs'][0]]
        #             residual_conv['params']['concat'] = prev_op['inputs'][1]
        #             consumed_op_names.add(prev_op['name'])
        #         else:
        #             residual_conv['inputs'] = [current_input_tensor]
        #     else:
        #         residual_conv['inputs'] = [current_input_tensor]

        #     # --- Trace forwards for post-operations ---
        #     current_output_tensor = op['outputs'][0]
            
        #     consumers = tensor_consumers.get(current_output_tensor, [])
        #     if len(consumers) == 1:
        #         next_op = consumers[0]
        #         if next_op and next_op['name'] not in consumed_op_names and next_op['type'] == 'Add':
        #             add_inputs = list(next_op['inputs'])
        #             add_inputs.remove(current_output_tensor)
        #             residual_conv['params']['add'] = add_inputs[0]
        #             consumed_op_names.add(next_op['name'])
        #             current_output_tensor = next_op['outputs'][0]

        #     consumers = tensor_consumers.get(current_output_tensor, [])
        #     if len(consumers) == 1:
        #         next_op = consumers[0]
        #         if next_op and next_op['name'] not in consumed_op_names and next_op['type'] == 'Relu':
        #             residual_conv['params']['relu'] = True
        #             consumed_op_names.add(next_op['name'])
        #             current_output_tensor = next_op['outputs'][0]

        #     consumers = tensor_consumers.get(current_output_tensor, [])
        #     if len(consumers) == 1:
        #         next_op = consumers[0]
        #         if next_op and next_op['name'] not in consumed_op_names and next_op['type'] == 'BasicPixelShuffle':
        #             residual_conv['params']['pixelShuffle2x2'] = True
        #             consumed_op_names.add(next_op['name'])
        #             current_output_tensor = next_op['outputs'][0]
            
        #     residual_conv['outputs'] = [current_output_tensor]
        #     consolidated_ops.append(residual_conv)

    consume_residual_group()

    # for op in ops_list:
    #     if op['name'] not in consumed_op_names:
    #         print(f"Warning: Operation '{op['name']}' of type '{op['type']}' was not consolidated.")

    # --- 5. Create the final JSON structure ---
    final_json = {
        'graph_inputs': [inp.name for inp in graph.input if inp.name not in initializers],
        'graph_outputs': [out.name for out in graph.output],
        'operations': consolidated_ops
    }

    # --- 6. Write files ---
    with open(weights_path, 'wb') as f:
        f.write(weights_blob)
    
    with open(json_path, 'w') as f:
        json.dump(final_json, f, indent=4, default=json_serializable)

    # --- 7. Print simplified graph representation ---
    print("\nSimplified Graph Representation:")
    print("-" * 40)
    
    output_to_idx = {}
    for i, op in enumerate(consolidated_ops):
        for out_name in op['outputs']:
            output_to_idx[out_name] = i + 1

    for i, op in enumerate(consolidated_ops):
        input_indices = [str(output_to_idx.get(inp, 0)) for inp in op['inputs']]
        inputs_str = ", ".join(input_indices)
        
        op_type_str = op['type']
        if op['type'] == 'ResidualConv':
            params = op.get('params', {})
            desc_parts = []
            if 'concat' in params:
                desc_parts.append(f"Concat={output_to_idx.get(params['concat'], 0)}")
            if params.get('pad') == 'clamp':
                desc_parts.append("Pad=Clamp")
            if 'add' in params:
                desc_parts.append(f"Add={output_to_idx.get(params['add'], 0)}")
            if params.get('relu', False):
                desc_parts.append("Relu")
            if params.get('pixelShuffle2x2', False):
                desc_parts.append("PixelShuffle2x2")
            
            if desc_parts:
                op_type_str = f"ResidualConv ({', '.join(desc_parts)})"

        print(f"{i+1}. [{op_type_str}] (Inputs: {inputs_str})")
    print("-" * 40)


    print(f"\nSuccessfully published ONNX model to:")
    print(f"  - Weights: {weights_path} ({len(weights_blob)} bytes)")
    print(f"  - JSON:    {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Publish an ONNX model to a custom format for Unity.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('onnx_file', help='Path to the input .onnx file.')
    parser.add_argument('--weights-out', required=True, help='Path to save the output weights blob.')
    parser.add_argument('--json-out', required=True, help='Path to save the output JSON definition.')
    
    args = parser.parse_args()
    
    publish_onnx(args.onnx_file, args.weights_out, args.json_out)

if __name__ == '__main__':
    main()