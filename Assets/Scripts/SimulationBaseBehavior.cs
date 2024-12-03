using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.VisualScripting;
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

        var output = new ComputeBuffer(data.Length, Marshal.SizeOf<T>(), ComputeBufferType.Structured, ComputeBufferMode.Immutable);
        output.SetData(data);
        DisposeOnDisable(output);
        return output;
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
                break;
            case ComputeBuffer buffer:
                shader.SetBuffer(kernelID, tuple.Item1, buffer);
                break;
            default:
                throw new Exception("What is " + tuple.Item1.GetType().Name + "?");
            }
        }

        shader.GetKernelThreadGroupSizes(kernelID, out var sizeX, out var sizeY, out var _);
        shader.Dispatch(kernelID, (int)((w - 1) / sizeX + 1), (int)((h - 1) / sizeY + 1), 1);
    }
}