using System;
using System.Collections.Generic;
using UnityEngine;

public abstract class DisposalHelperComponent : MonoBehaviour
{
    private List<IDisposable> disposeOnDisable = new List<IDisposable>();
    private List<IDisposable> disposeOnNextFrame = new List<IDisposable>();
    private List<IDisposable> disposeOnThisFrame = new List<IDisposable>();

    private class DisposableWrapper : IDisposable {
        public DisposableWrapper(Action onDispose) {
            _onDispose = onDispose;
        }

        Action _onDispose;
        public void Dispose() => _onDispose();
    }

    public void DisposeOnDisable(IDisposable o) {
        disposeOnDisable.Add(o);
    }

    public void DisposeOnDisable(Action disposal) {
        disposeOnDisable.Add(new DisposableWrapper(disposal));
    }

    public void DisposeOnNextFrame(IDisposable o)
    {
        disposeOnNextFrame.Add(o);
    }

    public void DisposeOnNextFrame(Action disposal)
    {
        disposeOnNextFrame.Add(new DisposableWrapper(disposal));
    }

    protected virtual void OnDisable() {
        foreach(var o in disposeOnDisable) {
            o.Dispose();
        }
        disposeOnDisable.Clear();
    }

    protected virtual void Update()
    {
        foreach(var o in disposeOnThisFrame) {
            o.Dispose();
        }
        disposeOnThisFrame.Clear();

        var swap = disposeOnThisFrame;
        disposeOnThisFrame = disposeOnNextFrame;
        disposeOnNextFrame = swap;
    }
}