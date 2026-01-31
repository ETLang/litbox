using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.U2D.IK;
using UnityEngine.UIElements;

public class CloudPuff : MonoBehaviour
{
    [SerializeField, Range(0,1), ReadOnly] float puffThickness = 1;

    [Header("Intermediates")]
    [SerializeField] bool manualConfig;
    [SerializeField] Color backgroundColor;
    [SerializeField] Color foregroundColor;
    [SerializeField] Color foregroundAmbient;
    [SerializeField] float obscurityStrength = 0.6f;

    [Header("Links")]
    [SerializeField] RTObject rayTracedObject;
    [SerializeField] Renderer backgroundObject;
    [SerializeField] Renderer foregroundObject;

    MaterialPropertyBlock _props;
    private static int _ColorId = Shader.PropertyToID("_Color");
    private static int _ForegroundColorId = Shader.PropertyToID("_ForegroundAmbientColor");
    private static int _ObscurityStrengthId = Shader.PropertyToID("_ObscurityStrength");


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        _props = new MaterialPropertyBlock();

        backgroundColor = backgroundObject.sharedMaterial.GetColor(_ColorId);
        foregroundColor = foregroundObject.sharedMaterial.GetColor(_ColorId);
        foregroundAmbient = foregroundObject.sharedMaterial.GetColor(_ForegroundColorId);
        obscurityStrength = foregroundObject.sharedMaterial.GetFloat(_ObscurityStrengthId);
    }

    // Update is called once per frame
    void Update()
    {
        if(!isActiveAndEnabled) return;

        if(!rayTracedObject) { return; 
        }

        if(manualConfig)
        {
            if(backgroundObject)
            {
                backgroundObject.GetPropertyBlock(_props);
                _props.SetColor(_ColorId, backgroundColor);
                backgroundObject.SetPropertyBlock(_props);
            }

            if(foregroundObject)
            {
                foregroundObject.GetPropertyBlock(_props);
                _props.SetColor(_ColorId, foregroundColor);
                _props.SetColor(_ForegroundColorId, foregroundAmbient);
                _props.SetFloat(_ObscurityStrengthId, obscurityStrength);
                foregroundObject.SetPropertyBlock(_props);
            }
        } else {
            // Link the alpha of the foreground and background to the density of the cloud
            var transmissibilityPerPixel = 1-Mathf.Pow(10, rayTracedObject.substrateLogDensity);
            var transmissibilityPerPuff = Mathf.Pow(transmissibilityPerPixel, transform.localScale.x * puffThickness);
            var puffDensity = 1-transmissibilityPerPuff;

            if(backgroundObject)
            {
                backgroundObject.GetPropertyBlock(_props);
                var c = backgroundColor;
                c.a = puffDensity;
                backgroundColor = c;
                _props.SetColor(_ColorId, c);
                backgroundObject.SetPropertyBlock(_props);
            }

            if(foregroundObject)
            {
                foregroundObject.GetPropertyBlock(_props);
                var c = foregroundColor;
                c.a = puffDensity;
                foregroundColor = c;
                obscurityStrength = 10 * puffThickness;// / Mathf.Sqrt(transmissibilityPerPuff);
                _props.SetColor(_ColorId, c);
                _props.SetFloat(_ObscurityStrengthId, obscurityStrength);
                foregroundObject.SetPropertyBlock(_props);
            }
        }
    }
}
