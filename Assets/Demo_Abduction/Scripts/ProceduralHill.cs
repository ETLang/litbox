using GLTFast;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
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

    CommandBuffer cb;
    MaterialPropertyBlock[] layerPropertyBlocks = new MaterialPropertyBlock[0];
    HashSet<Camera> registeredCameras = new HashSet<Camera>();
    PolygonCollider2D _collider;
    RTObject _rayTracing;
    int _layerListeningCount = 0;

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

            for (int i = 0; i < layers.Length; i++) {
                layerPropertyBlocks[i] = new MaterialPropertyBlock();
            }
        }

        for (int i = 0; i < layers.Length; i++) {
            var matrix = Matrix4x4.TRS(
                new Vector3(layers[i].textureOffset, 0, 0),
                Quaternion.AngleAxis(layers[i].textureAngle, Vector3.forward),
                new Vector3(layers[i].textureScale, layers[i].textureScale, 1));

            var density = _rayTracing.SubstrateDensity;

            layerPropertyBlocks[i].SetColor("_FuzzColor", layers[i].fuzzColor);
            layerPropertyBlocks[i].SetFloat("_FuzzLength", layers[i].fuzzLength);
            layerPropertyBlocks[i].SetColor("_SpecularColor", layers[i].specularColor * specularFilter);
            layerPropertyBlocks[i].SetFloat("_SpecularPower", layers[i].specularPower);

            layerPropertyBlocks[i].SetMatrix("_FarmlandTransform", matrix);
            layerPropertyBlocks[i].SetFloat("_FarmlandRowCount", layers[i].rowCount);
            layerPropertyBlocks[i].SetTexture("_MainTex", layers[i].hillTexture);
            layerPropertyBlocks[i].SetFloat("_ZOffset", 0);// i * 0.000001f);

            layerPropertyBlocks[i].SetFloat("_LeftHeight", leftHeight);
            layerPropertyBlocks[i].SetFloat("_PeakHeight", peakHeight);
            layerPropertyBlocks[i].SetFloat("_RightHeight", rightHeight);

            layerPropertyBlocks[i].SetVector("_SpecularSource", specularLightSource);
            layerPropertyBlocks[i].SetFloat("_ViewXShift", viewShift);
            layerPropertyBlocks[i].SetColor("_LeftAmbience", leftAmbience);
            layerPropertyBlocks[i].SetColor("_RightAmbience", rightAmbience);
            layerPropertyBlocks[i].SetColor("_Haze", haze);
            layerPropertyBlocks[i].SetFloat("_RayTracingVerticalOffset", rayTracingVerticalOffset);
        }
    }

    private void ListenOnArray(int index)
    {
        DetectChanges(() => index < layers.Length ? layers[index].fuzzColor : default(Color));
        DetectChanges(() => index < layers.Length ? layers[index].fuzzLength : default(float));
        DetectChanges(() => index < layers.Length ? layers[index].specularColor : default(Color));
        DetectChanges(() => index < layers.Length ? layers[index].specularPower : default(float));
        DetectChanges(() => index < layers.Length ? layers[index].hillTexture : default(Texture));
        DetectChanges(() => index < layers.Length ? layers[index].textureAngle : default(float));
        DetectChanges(() => index < layers.Length ? layers[index].textureOffset : default(float));
        DetectChanges(() => index < layers.Length ? layers[index].textureScale : default(float));
        DetectChanges(() => index < layers.Length ? layers[index].rowCount : default(int));
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

    protected override void OnInvalidated()
    {
        base.OnInvalidated();
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
        cb = new CommandBuffer();
        cb.name = gameObject.name + " CB";
    }

    void OnDisable()
    {
        // Always remember to remove the command buffer from the camera to avoid memory leaks.
        if (cb != null) {
            foreach (var cam in registeredCameras) {
                if (cam) {
                    cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
                }
            }
            registeredCameras.Clear();
            cb.Release();
            cb = null;
        }
    }

    protected override void OnDestroy()
    {
        base.OnDestroy();

        // Always remember to remove the command buffer from the camera to avoid memory leaks.
        if (cb != null) {
            foreach (var cam in registeredCameras) {
                if (cam) {
                    cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
                }
            }
            registeredCameras.Clear();
            cb.Release();
            cb = null;
        }
    }

    protected override void Update()
    {
        base.Update();

        List<Camera> toRemove = new List<Camera>();
        foreach(var cam in registeredCameras) {
            if((cam.cullingMask & (1 << gameObject.layer)) == 0) {
                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
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
                    cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
                    registeredCameras.Add(cam);
                }

#if UNITY_EDITOR
            var sceneCam = SceneView.lastActiveSceneView?.camera;
            if (sceneCam != null && !registeredCameras.Contains(sceneCam)) {
                sceneCam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
                registeredCameras.Add(sceneCam);
            }
#endif
        }

        var matrix = transform.localToWorldMatrix;

        cb.Clear();

#if UNITY_EDITOR
        var prefabStage = PrefabStageUtility.GetCurrentPrefabStage();

        if(prefabStage != null && !prefabStage.IsPartOfPrefabContents(gameObject)) {
            return;
        }
#endif

        for (int i = 0; i < layerPropertyBlocks.Length; i++) {
            layerPropertyBlocks[i].SetFloat("_substrateDensity", _rayTracing.SubstrateDensity);
            cb.DrawMesh(hillMesh, matrix, hillMat, 0, Math.Min(i, 1), layerPropertyBlocks[i]);
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
