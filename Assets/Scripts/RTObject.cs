using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class RTObject : MonoBehaviour
{
    [SerializeField] public Texture2D normal;

    [Range(-5,5)]
    [SerializeField] public float substrateLogDensity;

    [Range(0,1)]
    [SerializeField] public float objectHeight;

    public virtual Matrix4x4 WorldTransform
    {
        get => transform.localToWorldMatrix;
    }

    public bool Changed {get; protected set;}

    private Matrix4x4 _previousMatrix;
    private Texture2D _previousNormal;
    private float _previousSubstrateLogDensity;
    private float _previousObjectHeight;
    private bool _externallyInvalidated;

    protected void Start() {
        var renderer = GetComponent<Renderer>();
        
        _previousMatrix = WorldTransform;
        _previousNormal = normal;
        _previousSubstrateLogDensity = substrateLogDensity;
        _previousObjectHeight = objectHeight;

        var mat = renderer.material;
        mat.SetFloat("_substrateDensity", Mathf.Pow(10, substrateLogDensity));
    }
    
    protected void Update()
    {
        var renderer = GetComponent<Renderer>();

        Changed =
            _externallyInvalidated ||
            _previousMatrix != WorldTransform || 
            _previousNormal != normal ||
            _previousSubstrateLogDensity != substrateLogDensity || 
            _previousObjectHeight != objectHeight;

        _previousMatrix = WorldTransform;
        _previousNormal = normal;
        _previousSubstrateLogDensity = substrateLogDensity;
        _previousObjectHeight = objectHeight;

        if(Changed) {
            var mat = renderer.material;
            mat.SetFloat("_substrateDensity", Mathf.Pow(10, substrateLogDensity));
        }

        _externallyInvalidated = false;
    }

    public void Invalidate() {
        _externallyInvalidated = true;
    }
}
