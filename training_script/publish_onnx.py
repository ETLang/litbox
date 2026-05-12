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

    # --- 3. Create the final JSON structure ---
    final_json = {
        'graph_inputs': [inp.name for inp in graph.input if inp.name not in initializers],
        'graph_outputs': [out.name for out in graph.output],
        'operations': ops_list
    }

    # --- 4. Write files ---
    with open(weights_path, 'wb') as f:
        f.write(weights_blob)
    
    with open(json_path, 'w') as f:
        json.dump(final_json, f, indent=4, default=json_serializable)

    # --- 5. Print simplified graph representation ---
    print("\nSimplified Graph Representation:")
    print("-" * 40)
    
    # Map output names to the index of the operation that produced them
    output_to_idx = {}
    for i, op in enumerate(ops_list):
        for out_name in op['outputs']:
            output_to_idx[out_name] = i + 1

    for i, op in enumerate(ops_list):
        # Map input names to operation indices (0 if it's a graph input)
        input_indices = []
        for inp in op['inputs']:
            idx = output_to_idx.get(inp, 0)
            input_indices.append(str(idx))
        
        inputs_str = ", ".join(input_indices)
        print(f"{i+1}. [{op['type']}] (Inputs: {inputs_str})")
    print("-" * 40)


    print(f"Successfully published ONNX model to:")
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