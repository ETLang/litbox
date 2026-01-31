using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RTPointLight : RTLightSource
{
    [Range(0,1)]
    public float emissionOutscatter = 0.1f;

    public override Matrix4x4 WorldTransform
    {
        get => transform.localToWorldMatrix * Matrix4x4.Scale(new Vector3(0.5f, 0.5f, 1));
    }

    float _previousEmissionOutscatter;

    protected new void Start() {
        base.Start();
        
        _previousEmissionOutscatter = emissionOutscatter;
    }

    // Update is called once per frame
    protected new void Update()
    {
        base.Update();

        Changed = Changed || _previousEmissionOutscatter != emissionOutscatter;
        _previousEmissionOutscatter = emissionOutscatter;
    }
}
