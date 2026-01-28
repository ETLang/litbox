
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
#endif

    public bool DisposeOnExitingPlayMode { get; set; } = true;

    private List<IDisposable> toDispose = new List<IDisposable>();

    public Disposable()
    {
#if DETECT_LEAKS
        lock(_Allocations)
        {
            _allocationId = _NextId;
            _NextId++;
            _Allocations[_allocationId] = new StackTrace();
        }
#endif

#if UNITY_EDITOR
        UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
#endif
    }

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
#if UNITY_EDITOR
        UnityEditor.EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
#endif
        foreach(var o in toDispose) {
            o.Dispose();
        }
        toDispose.Clear();
    }

#if UNITY_EDITOR
    private void OnPlayModeStateChanged(UnityEditor.PlayModeStateChange state)
    {
        if(DisposeOnExitingPlayMode && state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
        {
            Dispose();
        }
    }
#endif
}