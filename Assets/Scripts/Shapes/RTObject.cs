using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RTObject : PhotonerComponent
{
    [SerializeField] public Texture texture;
    [SerializeField] public Texture normal;
    [SerializeField] public Color color = Color.white;
    [SerializeField, Range(-10,0)] public float substrateLogDensity;
    [SerializeField, Range(0, 1)] public float particleAlignment = 0;
    [SerializeField, Range(0, 1)] public float heightScale = 1;

    public virtual Matrix4x4 WorldTransform
    {
        get => transform.localToWorldMatrix;
    }

    public float SubstrateDensity => Mathf.Pow(10, substrateLogDensity);
    public bool Changed {get; protected set;}

    private MaterialPropertyBlock _propertyBlock;

    private static int _substrateDensityId = Shader.PropertyToID("_substrateDensity");
    private static int _particleAlignmentId = Shader.PropertyToID("_particleAlignment");
    private static int _colorId = Shader.PropertyToID("_Color");
    private static int _normalTexId = Shader.PropertyToID("_NormalTex");
    private static int _mainTexId = Shader.PropertyToID("_MainTex");
    private static int _heightScaleId = Shader.PropertyToID("_heightScale");

    protected virtual void Awake()
    {
    }

    protected virtual void Start() 
    {
        _propertyBlock = new MaterialPropertyBlock();

        //DetectChanges(() => WorldTransform);
        DetectChanges(() => normal);
        DetectChanges(() => substrateLogDensity);
        DetectChanges(() => particleAlignment);
        DetectChanges(() => heightScale);
        DetectChanges(() => texture);
        DetectChanges(() => color);
        DetectChanges(() => WorldTransform, "Transform");

        UpdateProperties();
    }

    protected override void Update()
    {
        Changed = false;
        base.Update();
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);

        Changed = true;
        
        if(group == "Transform") {
            return;
        }

        UpdateProperties();
    }

    void UpdateProperties() {
        var renderer = GetComponent<Renderer>();
        if(renderer != null) {
            //renderer.GetPropertyBlock(_propertyBlock);
            if(texture) {
                _propertyBlock.SetTexture(_mainTexId, texture);
            } else {
                _propertyBlock.SetTexture(_mainTexId, Texture2D.whiteTexture);
            }
            if(normal) {
                _propertyBlock.SetTexture(_normalTexId, normal);
            } else {
                _propertyBlock.SetTexture(_normalTexId, Texture2D.blackTexture);
            }
            _propertyBlock.SetColor(_colorId, color);
            _propertyBlock.SetFloat(_substrateDensityId, SubstrateDensity);
            _propertyBlock.SetFloat(_particleAlignmentId, particleAlignment);
            _propertyBlock.SetFloat(_heightScaleId, heightScale);
            renderer.SetPropertyBlock(_propertyBlock);
        }
    }
}
