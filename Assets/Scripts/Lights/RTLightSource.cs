using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public class RTLightSource : MonoBehaviour
{
    [Range(0, 80)]
    public float intensity = 1;
    public uint bounces = 2;

    public Vector4 Energy
    {
        get => (Vector4)gameObject.GetComponent<SpriteRenderer>().color * intensity;
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

    protected void Start() {
        _previousEnergy = Energy;
        _previousBounces = bounces;
        _previousMatrix = WorldTransform;
    }

    protected void Update()
    {
        Changed = (_previousEnergy != Energy) || (_previousBounces != bounces) || (_previousMatrix != WorldTransform);
        _previousEnergy = Energy;
        _previousBounces = bounces;
        _previousMatrix = WorldTransform;
    }
}