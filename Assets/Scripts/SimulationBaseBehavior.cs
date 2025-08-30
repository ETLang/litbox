using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;

public class SimulationBaseBehavior : MonoBehaviour {

    // DISPOSAL HELPERS
    private List<IDisposable> disposeOnDisable = new List<IDisposable>();

    private class DisposableWrapper : IDisposable {
        public DisposableWrapper(Action onDispose) {
            _onDispose = onDispose;
        }

        Action _onDispose;
        public void Dispose() => _onDispose();
    }

    protected void DisposeOnDisable(IDisposable o) {
        disposeOnDisable.Add(o);
    }

    protected void DisposeOnDisable(Action disposal) {
        disposeOnDisable.Add(new DisposableWrapper(disposal));
    }

    protected virtual void OnDisable() {
        foreach(var o in disposeOnDisable) {
            o.Dispose();
        }
        disposeOnDisable.Clear();
    }

    // BUFFER HELPERS
    protected RenderTexture CreateRWTexture(int w, int h, RenderTextureFormat format) {
        var output = new RenderTexture(w, h, 0, format)
        {
            enableRandomWrite = true,
            useMipMap = false,
        };
        output.Create();
        DisposeOnDisable(() => DestroyImmediate(output));
        return output;
    }

    protected RenderTexture CreateRWTextureWithMips(int w, int h, RenderTextureFormat format) {
        var output = new RenderTexture(w, h, 0, format)
        {
            enableRandomWrite = true,
            useMipMap = true,
            autoGenerateMips = false,
        };
        output.Create();
        output.GenerateMips();
        DisposeOnDisable(() => DestroyImmediate(output));
        return output;
    }

    protected ComputeBuffer CreateStructuredBuffer<T>(T[] data) {
        var structureSize = Marshal.SizeOf<T>();
        if(structureSize % 16 != 0) {
            throw new Exception($"{typeof(T).Name}'s size is {structureSize}. It must be a multiple of 16");
        }

        var output = new ComputeBuffer(data.Length, structureSize, ComputeBufferType.Structured, ComputeBufferMode.Immutable);
        output.SetData(data);
        DisposeOnDisable(output);
        return output;
    }

    protected GraphicsBuffer CreateRWStructuredBuffer<T>(int elements)
    {
        var structureSize = Marshal.SizeOf<T>();
        return new GraphicsBuffer(GraphicsBuffer.Target.Structured, elements, structureSize);
    }

    protected void Visualize(Texture2D target, IEnumerable<float> data, bool normalize=true) {
        var size = target.width * target.height;

        var min = data.Min();
        var max = data.Max();
        var med = data.OrderBy(k => k).ElementAt(size / 2);
        var um = (med - min) / (max - min);
        var p = 1;//Mathf.Log(0.5f) / Mathf.Log(um);

        if(normalize) {        
            target.SetPixels(data.Select(x => {
                var u = Mathf.Pow((x - min) / (max - min), p);
                return new Color(u,u,u,1);
            }).ToArray());
        } else {
            target.SetPixels(data.Select(x => new Color(x,x,x,1)).ToArray());
        }

        target.Apply();
    }

    // COMPUTE SHADER HELPERS
    protected void RunKernel(ComputeShader shader, string kernel, int n, params (string,object)[] args) {
        RunKernel(shader, kernel, n, 1, args);
    }

    protected void RunKernel(ComputeShader shader, string kernel, int w, int h, params (string,object)[] args) {
        var kernelID = shader.FindKernel(kernel);

        foreach(var tuple in args) {
            switch(tuple.Item2) {
            case Texture texture:
                shader.SetTexture(kernelID, tuple.Item1, texture);
                shader.SetVector($"lut_window_{tuple.Item1}", new Vector4(
                    0.5f / texture.width, 1 - 1.0f / texture.width,
                    0.5f / texture.height, 1 - 1.0f / texture.height));

                if(texture is Texture3D tex3D) {
                    shader.SetVector($"lut_slice_window_{tuple.Item1}", new Vector2(
                        0.5f / tex3D.depth, 1 - 1.0f / tex3D.depth));
                }
                break;
            case ComputeBuffer buffer:
                shader.SetBuffer(kernelID, tuple.Item1, buffer);
                break;
            case GraphicsBuffer gb:
                shader.SetBuffer(kernelID, tuple.Item1, gb);
                break;
            default:
                throw new Exception("What is " + tuple.Item1.GetType().Name + "?");
            }
        }

        shader.GetKernelThreadGroupSizes(kernelID, out var sizeX, out var sizeY, out var _);
        shader.Dispatch(kernelID, (int)((w - 1) / sizeX + 1), (int)((h - 1) / sizeY + 1), 1);
    }

    protected void SetShaderFlag(ComputeShader shader, string keyword, bool value) {
        var id = shader.keywordSpace.FindKeyword(keyword);
        shader.SetKeyword(id, value);
    }
}