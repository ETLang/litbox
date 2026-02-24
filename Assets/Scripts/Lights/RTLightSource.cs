using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public class RTLightSource : LitboxComponent
{
    [Range(0, 3)]
    public float intensity = 1;
    public uint bounces = 2;

    public Vector4 Energy
    {
        get => (Vector4)gameObject.GetComponent<SpriteRenderer>().color * intensity * intensity;
    }

    public virtual Matrix4x4 WorldTransform
    {
        get => transform.localToWorldMatrix;
    }

    public bool Changed {get; protected set;}
    private Vector4 _previousEnergy;
    private uint _previousBounces;
    private Color _previousColor;
    private Matrix4x4 _previousMatrix;

    protected virtual void Start() {
        DetectChanges(() => Energy);
        DetectChanges(() => bounces);
        DetectChanges(() => WorldTransform);
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);
        Changed = true;
    }

    protected override void Update()
    {
        base.Update();
    }
}