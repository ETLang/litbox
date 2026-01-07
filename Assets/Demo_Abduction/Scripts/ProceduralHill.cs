using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;


[Serializable]
public class HillLayerProperties
{
    [Header("Material")]
    public Color fuzzColor;
    public float fuzzLength;
    public Color specularColor;
    public float specularPower;

    [Header("Farmland")]
    public Texture hillTexture;
    public float textureAngle;
    public float textureOffset;
    public float textureScale;
    public int rowCount;
}

[ExecuteInEditMode]
[RequireComponent(typeof(PolygonCollider2D))]
public class ProceduralHill : PhotonerDemoComponent
{
    [Header("Hill Shape")]
    [SerializeField] Mesh hillMesh;
    [SerializeField] float leftHeight;
    [SerializeField] float peakHeight;
    [SerializeField] float rightHeight;

    [Header("Layers")]
    [SerializeField] public HillLayerProperties[] layers;

    [Header("Scene Context")]
    [SerializeField] Material hillMat;
    [SerializeField] Vector3 specularLightSource;
    [SerializeField] float viewShift;
    [SerializeField] public Color leftAmbience;
    [SerializeField] public Color rightAmbience;
    [SerializeField] public Color specularFilter = Color.white;
    [SerializeField] public Color haze;
    [SerializeField] public float rayTracingVerticalOffset = -0.1f;

    CommandBuffer _litCB;
    CommandBuffer _unlitCB;
    MaterialPropertyBlock[] layerPropertyBlocks = new MaterialPropertyBlock[0];
    MaterialPropertyBlock[] unlitLayerPropertyBlocks = new MaterialPropertyBlock[0];
    HashSet<Camera> registeredCameras = new HashSet<Camera>();
    PolygonCollider2D _collider;
    RTObject _rayTracing;
    int _layerListeningCount = 0;
    Texture _hdrLightMap;
    Matrix4x4 _simulationUVTransform;
    BindSimulationToCamera _binder;

    public ProceduralHill()
    {
        DetectChanges(() => layers);
        DetectChanges(() => leftHeight);
        DetectChanges(() => peakHeight);
        DetectChanges(() => rightHeight);
        DetectChanges(() => specularLightSource);
        DetectChanges(() => viewShift);
        DetectChanges(() => leftAmbience);
        DetectChanges(() => rightAmbience);
        DetectChanges(() => specularFilter);
        DetectChanges(() => haze);
        DetectChanges(() => rayTracingVerticalOffset);
        DetectChanges(() => _hdrLightMap);
        DetectChanges(() => _simulationUVTransform);
    }

    private void ValidateArrayListeners()
    {
        if(_rayTracing == null) {
            _rayTracing = GetComponent<RTObject>();
        }

        if (layers.Length > _layerListeningCount) {
            for (int i = _layerListeningCount; i < layers.Length; i++) {
                ListenOnArray(i);
            }
            _layerListeningCount = layers.Length;
        }

        if (layers.Length != layerPropertyBlocks.Length) {
            layerPropertyBlocks = new MaterialPropertyBlock[layers.Length];
            unlitLayerPropertyBlocks = new MaterialPropertyBlock[layers.Length];

            for (int i = 0; i < layers.Length; i++) {
                layerPropertyBlocks[i] = new MaterialPropertyBlock();
                unlitLayerPropertyBlocks[i] = new MaterialPropertyBlock();}
        }

        for (int i = 0; i < layers.Length; i++) {
            BuildLayerPropertyBlock(layerPropertyBlocks[i], layers[i], false);
            BuildLayerPropertyBlock(unlitLayerPropertyBlocks[i], layers[i], true);
        }
    }

    private void BuildLayerPropertyBlock(MaterialPropertyBlock block, HillLayerProperties layer, bool unlit = false)
    {
        var matrix = Matrix4x4.TRS(
            new Vector3(layer.textureOffset, 0, 0),
            Quaternion.AngleAxis(layer.textureAngle, Vector3.forward),
            new Vector3(layer.textureScale, layer.textureScale, 1));

        var density = _rayTracing.SubstrateDensity;

        block.SetColor("_FuzzColor", layer.fuzzColor);
        block.SetFloat("_FuzzLength", layer.fuzzLength);

        block.SetMatrix("_FarmlandTransform", matrix);
        block.SetFloat("_FarmlandRowCount", layer.rowCount);
        block.SetTexture("_MainTex", layer.hillTexture);
        block.SetFloat("_ZOffset", 0);// i * 0.000001f);

        block.SetFloat("_LeftHeight", leftHeight);
        block.SetFloat("_PeakHeight", peakHeight);
        block.SetFloat("_RightHeight", rightHeight);

        block.SetFloat("_RayTracingVerticalOffset", rayTracingVerticalOffset);
        block.SetColor("_Haze", haze);
        block.SetFloat("_ViewXShift", viewShift);
        block.SetVector("_SpecularSource", specularLightSource);
        block.SetFloat("_SpecularPower", layer.specularPower);

        if(unlit)
        {
            block.SetTexture("_diffuseLightMap", Texture2D.blackTexture);
            block.SetColor("_SpecularColor", Color.clear);
            block.SetColor("_LeftAmbience", Color.white);
            block.SetColor("_RightAmbience", Color.white);
        } else {
            if(_hdrLightMap != null) {
                block.SetTexture("_diffuseLightMap", _hdrLightMap); 
            }

            block.SetMatrix("_LightingUVTransform", _simulationUVTransform);
            block.SetColor("_SpecularColor", layer.specularColor * specularFilter);
            block.SetColor("_LeftAmbience", leftAmbience);
            block.SetColor("_RightAmbience", rightAmbience);
        }
    }

    private void ListenOnArray(int index)
    {
        DetectChanges(() => index < layers.Length ? layers[index].fuzzColor : default);
        DetectChanges(() => index < layers.Length ? layers[index].fuzzLength : default);
        DetectChanges(() => index < layers.Length ? layers[index].specularColor : default);
        DetectChanges(() => index < layers.Length ? layers[index].specularPower : default);
        DetectChanges(() => index < layers.Length ? layers[index].hillTexture : default);
        DetectChanges(() => index < layers.Length ? layers[index].textureAngle : default);
        DetectChanges(() => index < layers.Length ? layers[index].textureOffset : default);
        DetectChanges(() => index < layers.Length ? layers[index].textureScale : default);
        DetectChanges(() => index < layers.Length ? layers[index].rowCount : default);
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        ValidateArrayListeners();

        if (Application.isPlaying) {
            // Configure the collider
            _collider = GetComponent<PolygonCollider2D>();

            var pts3d = GenerateBoundary();
            var pts = pts3d.Select(pt => (Vector2)pt).ToArray();
            _collider.points = pts;
        }
    }

    protected override void OnInvalidated(string group)
    {
        base.OnInvalidated(group);
        ValidateArrayListeners();
    }

    void OnEnable()
    {
        // Check for required assets.
        if (hillMesh == null || hillMat == null) {
            Debug.LogError("Mesh or Material not assigned. Disabling.");
            this.enabled = false;
            return;
        }

        // Create a new command buffer instance.
        _litCB = new CommandBuffer();
        _litCB.name = gameObject.name + " CB";

        _unlitCB = new CommandBuffer();
        _unlitCB.name = gameObject.name + " Unlit CB";
    }

    void Cleanup()
    {
        if (_litCB != null) {
            foreach (var cam in registeredCameras) {
                if (cam) {
                    cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, _litCB);
                }
            }
            _litCB.Release();
            _litCB = null;
        }

        if (_unlitCB != null) {
            foreach (var cam in registeredCameras) {
                if (cam) {
                    cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, _unlitCB);
                }
            }
            _unlitCB.Release();
            _unlitCB = null;
        }
        registeredCameras.Clear();
    }

    protected override void OnDisable()
    {
        base.OnDisable();
        Cleanup();
    }

    protected override void OnDestroy()
    {
        base.OnDestroy();
        Cleanup();
    }

    protected override void Update()
    {
        base.Update();

        if(_binder == null) {
            _binder = Camera.main.GetComponentInChildren<BindSimulationToCamera>();
        }

        if(_binder)
        {
            _hdrLightMap = _binder.GetComponent<Simulation>().SimulationOutputHDR;
            _simulationUVTransform = _binder.ScreenToSimulationUVTransform;
        }

        List<Camera> toRemove = new List<Camera>();
        foreach(var cam in registeredCameras) {
            if(!cam) {
                toRemove.Add(cam);
            } else if((cam.cullingMask & (1 << gameObject.layer)) == 0) {
                if(cam.GetComponent<SimulationCamera>() != null) {
                    cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, _unlitCB);
                } else {
                    cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, _litCB);
                }
                toRemove.Add(cam);
            }
        }

        foreach(var cam in toRemove) {
            registeredCameras.Remove(cam);
        }

        if (enabled) {
            var allCameras = FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            foreach (var cam in allCameras)
                if ((cam.cullingMask & (1 << gameObject.layer)) != 0 && !registeredCameras.Contains(cam)) {
                    if(cam.GetComponent<SimulationCamera>() != null) {
                        cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, _unlitCB);
                    } else {
                        cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, _litCB);
                    }
                    registeredCameras.Add(cam);
                }

#if UNITY_EDITOR
            var sceneCam = SceneView.lastActiveSceneView?.camera;
            if (sceneCam != null && !registeredCameras.Contains(sceneCam)) {
                sceneCam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, _litCB);
                registeredCameras.Add(sceneCam);
            }
#endif
        }

        var matrix = transform.localToWorldMatrix;

        _litCB.Clear();
        _unlitCB.Clear();

#if UNITY_EDITOR
        var prefabStage = PrefabStageUtility.GetCurrentPrefabStage();

        if(prefabStage != null && !prefabStage.IsPartOfPrefabContents(gameObject)) {
            return;
        }
#endif

        for (int i = 0; i < layerPropertyBlocks.Length; i++) {
            layerPropertyBlocks[i].SetFloat("_substrateDensity", _rayTracing.SubstrateDensity);
            unlitLayerPropertyBlocks[i].SetFloat("_substrateDensity", _rayTracing.SubstrateDensity);
            _litCB.DrawMesh(hillMesh, matrix, hillMat, 0, Math.Min(i, 1), layerPropertyBlocks[i]);

            if(i == 0)
                _unlitCB.DrawMesh(hillMesh, matrix, hillMat, 0, Math.Min(i, 1), unlitLayerPropertyBlocks[i]);
        }
    }

    Vector3 ComputeHillVertex(float u, float v)
    {
        const float perspective = 2;
        const float xLeft = -5;
        const float xRight = 5;
        const float gaussianLimit = 2;
        float gaussianLowerbound = Mathf.Exp(-Mathf.Pow(gaussianLimit, 2));

        float x = Mathf.Lerp(xLeft, xRight, u) * Mathf.Lerp(perspective, 1, 1 - Mathf.Pow(1 - v, 2)); ;
        float wx = Mathf.Lerp(-gaussianLimit, gaussianLimit, u);
        float w = Mathf.Exp(-Mathf.Pow(wx, 2));
        float dy = -2 * wx * w;
        w = (w - gaussianLowerbound) / (1 - gaussianLowerbound);
        float top;

        if (u < 0.5) {
            top = Mathf.Lerp(leftHeight, peakHeight, w);
            dy *= (peakHeight - leftHeight);
        } else {
            top = Mathf.Lerp(rightHeight, peakHeight, w);
            dy *= (peakHeight - rightHeight);
        }
        float y = top * (Mathf.Exp(-Mathf.Pow(1 - v, 2)) - Mathf.Exp(-1)) / (1 - Mathf.Exp(-1)); // v;

        return new Vector3(x, y, 0);
    }

    Vector3[] GenerateBoundary()
    {
        const int edgePts = 50;
        Gizmos.color = new Color(1, 1, 1, 1);
        var pts = new Vector3[4 * edgePts - 3];

        int i = 0;
        for (; i < edgePts; i++) {
            float u = i / (float)(edgePts - 1);
            pts[i] = ComputeHillVertex(u, 0);
        }
        for (; i < 2 * edgePts - 1; i++) {
            float v = (i - (edgePts - 1)) / (float)(edgePts - 1);
            pts[i] = ComputeHillVertex(1, v);
        }
        for (; i < 3 * edgePts - 2; i++) {
            float u = 1 - (i - (2 * edgePts - 2)) / (float)(edgePts - 1);
            pts[i] = ComputeHillVertex(u, 1);
        }
        for (; i < 4 * edgePts - 3; i++) {
            float v = 1 - (i - (3 * edgePts - 3)) / (float)(edgePts - 1);
            pts[i] = ComputeHillVertex(0, v);
        }
        return pts;
    }

#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        Gizmos.color = new Color(1, 1, 1, 0);
        var yLocalCenter = peakHeight / 2.0f;
        var worldMat = transform.localToWorldMatrix;

        var position = worldMat.MultiplyPoint(new Vector3(0, yLocalCenter, 0));
        Gizmos.DrawCube(position, new Vector3(transform.localScale.x * 5, peakHeight, 1));

        if (UnityEditor.Selection.Contains(gameObject)) {
            var pts = GenerateBoundary();
            for(int i = 0;i < pts.Length;i++) {
                pts[i] = worldMat.MultiplyPoint(pts[i]);
            }

            Gizmos.DrawLineStrip(pts, true);
        }
    }
#endif
}
