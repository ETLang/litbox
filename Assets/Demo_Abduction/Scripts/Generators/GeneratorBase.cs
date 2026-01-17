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

public class GeneratorBase<T> : PhotonerComponent, IProceduralGenerator where T : Object
{
    public GeneratorBase() : base(false) { }

    public virtual Object asset { get; }

    Type IProceduralGenerator.assetType => typeof(T);

    public event Action<IProceduralGenerator> Invalidated;

    public virtual void Populate(Object asset) { }

    public virtual Object CreateAsset() => (Object)Activator.CreateInstance(typeof(T));

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);

        Invalidated?.Invoke(this);
    }
}
