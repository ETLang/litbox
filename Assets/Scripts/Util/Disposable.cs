
#define DETECT_LEAKS

using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

public class Disposable : IDisposable
{
#if DETECT_LEAKS
    private static Dictionary<int,StackTrace> _Allocations = new Dictionary<int, StackTrace>();
    private static int _NextId = 1;
    private int _allocationId;

    public Disposable()
    {
        lock(_Allocations)
        {
            _allocationId = _NextId;
            _NextId++;
            _Allocations[_allocationId] = new StackTrace();
        }
    }
#endif

    private List<IDisposable> toDispose = new List<IDisposable>();

    private class DisposableWrapper : IDisposable {
        public DisposableWrapper(Action onDispose) {
            _onDispose = onDispose;
        }

        Action _onDispose;
        public void Dispose() => _onDispose();
    }

    public void AutoDispose(IDisposable o) {
        toDispose.Add(o);
    }

    public void AutoDispose(Action disposal) {
        toDispose.Add(new DisposableWrapper(disposal));
    }

    bool disposed;
    public void Dispose()
    {
        if(disposed) return;
        disposed = true;
        OnDispose();
    }

    ~Disposable()
    {
        if(!disposed)
        {
            UnityEngine.Debug.LogError("Failed to Dispose() something that needs to be disposed!");

#if DETECT_LEAKS
            var trace = _Allocations[_allocationId];
            UnityEngine.Debug.LogError(trace);
#endif
        }
    }

    protected virtual void OnDispose() 
    {
        foreach(var o in toDispose) {
            o.Dispose();
        }
        toDispose.Clear();
    }
}