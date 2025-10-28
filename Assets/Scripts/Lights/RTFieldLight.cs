using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RTFieldLight : RTLightSource
{
    [Range(0,1)]
    public float emissionOutscatter = 0.1f;

    public Texture2D lightTexture;

    public override Matrix4x4 WorldTransform
    {
        get => transform.localToWorldMatrix * Matrix4x4.Scale(new Vector3(0.5f, 0.5f, 1));
    }

    float _previousEmissionOutscatter;
    Texture2D _previousLightTexture;

    protected new void Start() {
        base.Start();

        _previousEmissionOutscatter = emissionOutscatter;
        _previousLightTexture = lightTexture;
    }

    // Update is called once per frame
    protected new void Update()
    {
        base.Update();

        Changed = Changed ||
            _previousEmissionOutscatter != emissionOutscatter ||
            _previousLightTexture != lightTexture;

        _previousEmissionOutscatter = emissionOutscatter;
        _previousLightTexture = lightTexture;
    }
}
