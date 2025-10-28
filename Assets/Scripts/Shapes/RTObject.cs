using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RTObject : MonoBehaviour
{
    [SerializeField] public Texture2D normal;

    [Range(-10,1)]
    [SerializeField] public float substrateLogDensity;

    [Range(0,1)]
    [SerializeField] public float objectHeight;

    public virtual Matrix4x4 WorldTransform
    {
        get => transform.localToWorldMatrix;
    }

    public float SubstrateDensity => Mathf.Pow(10, substrateLogDensity);
    public bool Changed {get; protected set;}

    private Matrix4x4 _previousMatrix;
    private Texture2D _previousNormal;
    private Material _mat;
    private float _previousSubstrateLogDensity;
    private float _previousObjectHeight;
    private bool _externallyInvalidated;

    private static int _substrateDensityId = Shader.PropertyToID("_substrateDensity");

    protected virtual void Awake()
    {

    }

    protected virtual void Start() {
        var renderer = GetComponent<Renderer>();
        if (renderer != null) {
            _mat = renderer.material;
        }

        _previousMatrix = WorldTransform;
        _previousNormal = normal;
        _previousSubstrateLogDensity = substrateLogDensity;
        _previousObjectHeight = objectHeight;

        _mat?.SetFloat(_substrateDensityId, SubstrateDensity);
    }
    
    protected virtual void Update()
    {
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
            var renderer = GetComponent<Renderer>();
            if (renderer != null) {
                _mat = renderer.material;
            }
            _mat?.SetFloat(_substrateDensityId, SubstrateDensity);
        }

        _externallyInvalidated = false;
    }

    public void Invalidate() {
        _externallyInvalidated = true;
    }
}
