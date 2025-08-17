using System.Collections.Generic;
using System;
using UnityEngine;
using Object = UnityEngine.Object;

public interface IProceduralGenerator
{
    void Populate(Object asset);
    bool CheckChanged();
    Object CreateAsset();
    Type assetType { get; }
    Object asset { get; }
    GameObject gameObject { get; }
    event Action<IProceduralGenerator> Invalidated;
}

public interface IChangeManager
{
    bool Check();
}

public class ChangeManager<T> : IChangeManager
{
    private T _previous = default(T);
    private Func<T> _getter;
    private Action _invalidate;

    public ChangeManager(Func<T> getter, Action invalidate)
    {
        _getter = getter;
        _previous = getter();
        _invalidate = invalidate;
    }

    public bool Check()
    {
        bool changed = false;

        var current = _getter();
        var c_null = current == null;
        var p_null = _previous == null;

        if (c_null != p_null) {
            changed = true;
        } else if (!p_null) {
            changed = !_previous.Equals(current);
        }

        if (changed) {
            _previous = current;
            _invalidate();
        }

        return changed;
    }
}

public class GeneratorBase<T> : PhotonerDemoComponent, IProceduralGenerator where T : Object
{
    public GeneratorBase() : base(false) { }

    public virtual Object asset { get; }

    Type IProceduralGenerator.assetType => typeof(T);

    public event Action<IProceduralGenerator> Invalidated;

    public virtual void Populate(Object asset) { }

    public virtual Object CreateAsset() => (Object)Activator.CreateInstance(typeof(T));

    protected override void OnInvalidated()
    {
        base.OnInvalidated();

        Invalidated?.Invoke(this);
    }
}
