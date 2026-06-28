using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RTSpotLight : RTLightSource
{
    [SerializeField, Range(0.01f,10)] public float pinch = 0.01f;

    protected override void Start()
    {
        base.Start();
        DetectChanges(() => pinch);
    }
}
