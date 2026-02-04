using UnityEngine;
using UnityEditor;
using System;

[System.Serializable]
public class SubstrateElementTransform {
    public Vector2 position;
    public Vector2 scale;
    [Range(0, 360)] public float angle;

    public Matrix4x4 matrix => Matrix4x4.TRS((Vector3)position, Quaternion.AngleAxis(angle, new Vector3(0,0,1)), new Vector3(scale.x, scale.y, 1));
}

[RequireComponent(typeof(Renderer))]
[RequireComponent(typeof(RTObject))]
public class TrainingSubstrate : MonoBehaviour {
    public Texture generated {get;private set;}
    public int textureSize = 512;
    [ReadOnly, HexInt(8)] public uint seed;
    public SubstrateElementTransform[] rects = new SubstrateElementTransform[] { 
        new SubstrateElementTransform {
            position = new Vector2(0.2f, 0.2f),
            scale = new Vector2(0.3f, 0.3f),
            angle = 30
        }
    };

    public SubstrateElementTransform[] ellipses = new SubstrateElementTransform[] {
        new SubstrateElementTransform {
            position = new Vector2(-0.1f, 0.2f),
            scale = new Vector2(0.4f, 0.2f),
            angle = 5
        }
    };

    public SubstrateElementTransform[] inverseRects = new SubstrateElementTransform[0];
    public SubstrateElementTransform[] inverseEllipses = new SubstrateElementTransform[0];
    [Range(1,256)] public float edgeBlur = 10;
    [Range(-1,1)] public float sharpness = 0;

    [Header("Noise")]
    public bool hasNoise;
    [Range(0,9)] public int minNoiseLevel = 0;
    [Range(0,9)] public int maxNoiseLevel = 0;
    [Range(0,1)] public float noiseFloor = 0;
    [Range(0,1)] public float noiseCeiling = 1;

    [Header("Color")]
    public Color colorA = Color.white;
    public Color colorB = Color.white;
    [Range(0,1)] public float densityA = 0.1f;
    [Range(0,1)] public float densityB = 0.01f;
    [Range(0,360)] public float gradientAngle = 90;
    [Range(0.1f, 1.4f)] public float gradientLength = 0.7f;

    void Start() {
        ValidateAndApply();
        GetComponent<Renderer>().material.SetTexture("_MainTex", generated);
    }

    public void GenerateRandom(int version=1) {
        GenerateRandom((uint)new System.Random(Environment.TickCount).Next(), version);
    }

    public void GenerateRandom(uint seed, int version=1) {
        var rand = new System.Random((int)seed);

        this.seed = seed;

        rects = new SubstrateElementTransform[rand.Next(4)];
        ellipses = new SubstrateElementTransform[rand.Next(4)];
        inverseRects = new SubstrateElementTransform[rand.Next(3)];
        inverseEllipses = new SubstrateElementTransform[rand.Next(3)];

        if(rects.Length == 0 && ellipses.Length == 0)
            rects = new SubstrateElementTransform[1];

        for(int i = 0;i < rects.Length;i++) {
            rects[i] = new SubstrateElementTransform {
                position = new Vector2(rand.NextRange(-0.9f, 0.9f), rand.NextRange(-0.9f, 0.9f)),
                angle = rand.NextRange(0, 360.0f),
                scale = new Vector2(rand.NextRange(0.1f, 0.7f), rand.NextRange(0.1f, 0.7f)),
            };
        }
        for(int i = 0;i < ellipses.Length;i++) {
            ellipses[i] = new SubstrateElementTransform {
                position = new Vector2(rand.NextRange(-0.9f, 0.9f), rand.NextRange(-0.9f, 0.9f)),
                angle = rand.NextRange(0, 360.0f),
                scale = new Vector2(rand.NextRange(0.1f, 1), rand.NextRange(0.1f, 1)),
            };
        }
        for(int i = 0;i < inverseRects.Length;i++) {
            inverseRects[i] = new SubstrateElementTransform {
                position = new Vector2(rand.NextRange(-0.7f, 0.7f), rand.NextRange(-0.7f, 0.7f)),
                angle = rand.NextRange(0, 360.0f),
                scale = new Vector2(rand.NextRange(0.1f, 0.3f), rand.NextRange(0.1f, 0.3f)),
            };
        }
        for(int i = 0;i < inverseEllipses.Length;i++) {
            inverseEllipses[i] = new SubstrateElementTransform {
                position = new Vector2(rand.NextRange(-0.7f, 0.7f), rand.NextRange(-0.7f, 0.7f)),
                angle = rand.NextRange(0, 360.0f),
                scale = new Vector2(rand.NextRange(0.1f, 0.3f), rand.NextRange(0.1f, 0.3f)),
            };
        }

        edgeBlur = rand.NextRange(1.0f, 128.0f, 0.3f);
        sharpness = rand.NextRange(-1,1);
        hasNoise = rand.NextBool(0.75f);
        minNoiseLevel = rand.Next(6);
        maxNoiseLevel = minNoiseLevel + rand.Next(5);

        noiseFloor = rand.NextRange(0, 0.6f, 0.75f);
        noiseCeiling = rand.NextRange(0.6f, 1);

        colorA = Color.HSVToRGB(rand.NextSingle(), rand.NextRange(0,1,0.75f), rand.NextRange(0.25f, 1, 0.75f));
        colorB = Color.HSVToRGB(rand.NextSingle(), rand.NextRange(0,1,0.75f), rand.NextRange(0.25f, 1, 0.75f));
        densityA = rand.NextRange(0.01f, 0.99f);
        densityB = rand.NextRange(0.01f, 0.99f);
        // densityA = Mathf.Min(Mathf.Pow(10, rand.NextRange(-2, 0)), 0.99f);
        // densityB = Mathf.Min(Mathf.Pow(10, rand.NextRange(-2, 0)), 0.99f);
        
        gradientAngle = rand.NextRange(0, 360);
        gradientLength = rand.NextRange(0.1f, 1.4f);

        bool hasGradient = rand.NextBool();

        if(!hasGradient) {
            colorB = colorA;
            densityB = densityA;
        }

        if(version == 2) {
            minNoiseLevel = rand.Next(3);
            maxNoiseLevel = 5 + rand.Next(5);
            noiseFloor = rand.NextRange(0, 0.3f, 0.5f);
            noiseCeiling = rand.NextRange(0.85f, 1);
        }
    }

    public void ValidateAndApply() {
        bool changed = false;
        changed = changed || textureSize != _old_textureSize;
        changed = changed || edgeBlur != _old_edgeBlur;
        changed = changed || sharpness != _old_sharpness;
        changed = changed || hasNoise != _old_hasNoise;
        changed = changed || minNoiseLevel != _old_minNoiseLevel;
        changed = changed || maxNoiseLevel != _old_maxNoiseLevel;
        changed = changed || noiseFloor != _old_noiseFloor;
        changed = changed || noiseCeiling != _old_noiseCeiling;
        changed = changed || colorA != _old_colorA;
        changed = changed || colorB != _old_colorB;
        changed = changed || densityA != _old_densityA;
        changed = changed || densityB != _old_densityB;
        changed = changed || gradientAngle != _old_gradientAngle;
        changed = changed || gradientLength != _old_gradientLength;
        changed = changed || rects.Length != _old_rects.Length;
        changed = changed || ellipses.Length != _old_ellipses.Length;
        changed = changed || inverseRects.Length != _old_inverseRects.Length;
        changed = changed || inverseEllipses.Length != _old_inverseEllipses.Length;

        if(!changed) {
            for(int i = 0;i < rects.Length;i++) {
                changed = changed || rects[i].position != _old_rects[i].position;
                changed = changed || rects[i].angle != _old_rects[i].angle;
                changed = changed || rects[i].scale != _old_rects[i].scale;
            }
            for(int i = 0;i < ellipses.Length;i++) {
                changed = changed || ellipses[i].position != _old_ellipses[i].position;
                changed = changed || ellipses[i].angle != _old_ellipses[i].angle;
                changed = changed || ellipses[i].scale != _old_ellipses[i].scale;
            }
            for(int i = 0;i < inverseRects.Length;i++) {
                changed = changed || inverseRects[i].position != _old_inverseRects[i].position;
                changed = changed || inverseRects[i].angle != _old_inverseRects[i].angle;
                changed = changed || inverseRects[i].scale != _old_inverseRects[i].scale;
            }
            for(int i = 0;i < inverseEllipses.Length;i++) {
                changed = changed || inverseEllipses[i].position != _old_inverseEllipses[i].position;
                changed = changed || inverseEllipses[i].angle != _old_inverseEllipses[i].angle;
                changed = changed || inverseEllipses[i].scale != _old_inverseEllipses[i].scale;
            }
        }

        _old_textureSize = textureSize;
        _old_edgeBlur = edgeBlur;
        _old_sharpness = sharpness;
        _old_hasNoise = hasNoise;
        _old_minNoiseLevel = minNoiseLevel;
        _old_maxNoiseLevel = maxNoiseLevel;
        _old_noiseFloor = noiseFloor;
        _old_noiseCeiling = noiseCeiling;
        _old_colorA = colorA;
        _old_colorB = colorB;
        _old_densityA = densityA;
        _old_densityB = densityB;
        _old_gradientAngle = gradientAngle;
        _old_gradientLength = gradientLength;
        _old_rects = (SubstrateElementTransform[])rects.Clone();
        _old_ellipses = (SubstrateElementTransform[])ellipses.Clone();
        _old_inverseRects = (SubstrateElementTransform[])inverseRects.Clone();
        _old_inverseEllipses = (SubstrateElementTransform[])inverseEllipses.Clone();

        if(changed) {
            ForceCreateTexture();
            GetComponent<RTObject>().Invalidate();
        }
    }

    public Texture ForceCreateTexture() {
        int destTarget = 0;
        if(_target == null || _target.Length != 2 || _target[0].width != textureSize) {
            if(_target != null) {
                foreach(var target in _target)
                    DestroyImmediate(target);
            }

            _target = new RenderTexture[2];
            for(int i = 0;i < _target.Length;i++) {
                _target[i] = new RenderTexture(textureSize, textureSize, 0, RenderTextureFormat.ARGBFloat)
                {
                    enableRandomWrite = true
                };
                _target[i].Create();
            }
        }
        
        var shader = (ComputeShader)Resources.Load("TrainingSubstrateGenerators");

        var transforms = new Matrix4x4[rects.Length + ellipses.Length + inverseRects.Length + inverseEllipses.Length];

        {
            int outIndex = 0;
            foreach(var t in rects) {
                transforms[outIndex] = t.matrix.inverse.transpose;
                outIndex++;
            }
            foreach(var t in ellipses) {
                transforms[outIndex] = t.matrix.inverse.transpose;
                outIndex++;
            }
            foreach(var t in inverseRects) {
                transforms[outIndex] = t.matrix.inverse.transpose;
                outIndex++;
            }
            foreach(var t in inverseEllipses) {
                transforms[outIndex] = t.matrix.inverse.transpose;
                outIndex++;
            }
        }

        Vector3 gradient = new Vector3(
            Mathf.Cos(gradientAngle / 180.0f * Mathf.PI) / gradientLength, 
            Mathf.Sin(gradientAngle / 180.0f * Mathf.PI) / gradientLength, 
            gradientLength / 2.0f);


        shader.SetVector("g_trainingSubstrate_size", new Vector2(textureSize, textureSize));
        shader.SetVector("g_trainingSubstrate_ShapeCounts", new Vector4(rects.Length, ellipses.Length, inverseRects.Length, inverseEllipses.Length));
        shader.SetMatrixArray("g_trainingSubstrate_ShapeTransforms", transforms);
        shader.SetFloat("g_trainingSubstrate_edgeBlur", edgeBlur);
        shader.SetFloat("g_trainingSubstrate_hardness", Mathf.Pow(10, sharpness));
        shader.SetVector("g_trainingSubstrate_noiseSeed", new Vector2((int)((seed & 0xFFFF0000) >> 16), (int)(seed & 0x0000FFFF)));
        shader.SetVector("g_trainingSubstrate_noiseFrequencyMinMax", new Vector2((1 << minNoiseLevel), (1 << maxNoiseLevel)));
        shader.SetVector("g_trainingSubstrate_noiseClipValues", new Vector2(noiseFloor, noiseCeiling));
        shader.SetVector("g_trainingSubstrate_gradientDirection", gradient);
        shader.SetVector("g_trainingSubstrate_gradientDensity", new Vector2(densityA, densityB));
        shader.SetVector("g_trainingSubstrate_gradientColorA", colorA);
        shader.SetVector("g_trainingSubstrate_gradientColorB", colorB);

        int kernel;
        // Make shapes (rects and ellipses)
        kernel = shader.FindKernel("GenerateTrainingSubstrate_MakeShapes");
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", _target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", _target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Blur edge
        kernel = shader.FindKernel("GenerateTrainingSubstrate_EdgeBlur_JFA");
        for(int i = 1;i < textureSize;i *= 2) {
            shader.SetInt("g_trainingSubstrate_edgeBlurStage", i);
            destTarget = 1 - destTarget;
            shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", _target[1-destTarget]);
            shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", _target[destTarget]);
            shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);
        }

        kernel = shader.FindKernel("GenerateTrainingSubstrate_EdgeBlur");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", _target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", _target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Add Noise
        if(hasNoise) {
            kernel = shader.FindKernel("GenerateTrainingSubstrate_AddNoise");
            destTarget = 1 - destTarget;
            shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", _target[1-destTarget]);
            shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", _target[destTarget]);
            shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);
        }

        // Gradient/Coloration
        kernel = shader.FindKernel("GenerateTrainingSubstrate_Gradient");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", _target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", _target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        // Hardness
        kernel = shader.FindKernel("GenerateTrainingSubstrate_Hardness");
        destTarget = 1 - destTarget;
        shader.SetTexture(kernel, "g_trainingSubstrate_textureRef", _target[1-destTarget]);
        shader.SetTexture(kernel, "g_trainingSubstrate_textureDest", _target[destTarget]);
        shader.Dispatch(kernel, Math.Max(1, textureSize / 8), Math.Max(1, textureSize / 8), 1);

        generated = _target[destTarget];
#if UNITY_EDITOR
        if(EditorApplication.isPlaying)
             GetComponent<RTObject>().texture = _target[destTarget];
#endif

        return _target[1-destTarget];
    }

#region Private
    private int _old_textureSize;
    private SubstrateElementTransform[] _old_rects = new SubstrateElementTransform[0];
    private SubstrateElementTransform[] _old_ellipses = new SubstrateElementTransform[0];
    private SubstrateElementTransform[] _old_inverseRects = new SubstrateElementTransform[0];
    private SubstrateElementTransform[] _old_inverseEllipses = new SubstrateElementTransform[0];
    private float _old_edgeBlur;
    private float _old_sharpness;
    private bool _old_hasNoise;
    private int _old_minNoiseLevel;
    private int _old_maxNoiseLevel;
    private float _old_noiseFloor;
    private float _old_noiseCeiling;
    private Color _old_colorA;
    private Color _old_colorB;
    private float _old_densityA;
    private float _old_densityB;
    private float _old_gradientAngle;
    private float _old_gradientLength;

    private RenderTexture[] _target;
#endregion
}