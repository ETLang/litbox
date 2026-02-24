using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RTSpotLight : RTLightSource
{
    [SerializeField, Range(0.01f,10)] public float tightness;

    protected override void Start()
    {
        base.Start();

        DetectChanges(() => tightness);
    }
}
