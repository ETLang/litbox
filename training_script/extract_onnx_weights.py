import argparse
import csv
import onnx
from onnx import numpy_helper

def main():
    parser = argparse.ArgumentParser(description="Extract weights from an ONNX Conv layer to CSV.")
    parser.add_argument('--onnx', required=True, help="Path to the ONNX file")
    parser.add_argument('--csv', help="Path to the output CSV file")
    parser.add_argument('--layer', help="Name or index of the layer to extract from")
    parser.add_argument('--channel', type=int, help="Index of the output channel")
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    
    if args.layer is None or args.csv is None or args.channel is None:
        print(f"Listing layers in '{args.onnx}' (provide --layer, --channel, and --csv to extract):")
        for i, node in enumerate(model.graph.node):
            print(f"{i:4d}: {node.name} [{node.op_type}]")
        return

    target_node = None
    try:
        layer_idx = int(args.layer)
        if 0 <= layer_idx < len(model.graph.node):
            target_node = model.graph.node[layer_idx]
        else:
            print(f"Error: Layer index {layer_idx} is out of bounds.")
            return
    except ValueError:
        for node in model.graph.node:
            if node.name == args.layer or args.layer in node.output:
                target_node = node
                break
            
    if not target_node:
        print(f"Error: Layer '{args.layer}' not found in the ONNX graph.")
        return
        
    if target_node.op_type != 'Conv':
        print(f"Error: Layer '{args.layer}' is of type '{target_node.op_type}', expected 'Conv'.")
        return
        
    initializers = {init.name: init for init in model.graph.initializer}
    
    weight_name = target_node.input[1]
    if weight_name not in initializers:
        print(f"Error: Weight initializer '{weight_name}' not found.")
        return
        
    weights = numpy_helper.to_array(initializers[weight_name])
    
    bias = 0.0
    if len(target_node.input) > 2:
        bias_name = target_node.input[2]
        if bias_name in initializers:
            bias_array = numpy_helper.to_array(initializers[bias_name])
            if args.channel < len(bias_array):
                bias = float(bias_array[args.channel])
    
    if args.channel >= weights.shape[0]:
        print(f"Error: Channel index {args.channel} is out of bounds for weight shape {weights.shape}.")
        return
        
    target_weights = weights[args.channel]
    
    with open(args.csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([bias])
        writer.writerow([])
        
        for in_c in range(target_weights.shape[0]):
            for row in target_weights[in_c]:
                writer.writerow(row)
            writer.writerow([])

if __name__ == '__main__':
    main()