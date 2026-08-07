using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteAlways]
public class RTRect : RTObject
{

    MeshRenderer _meshRenderer;
    MeshFilter _meshFilter;
    Mesh _mesh;
    Material _rtMat;

    [SerializeField] Mesh mesh;

    protected override void Awake()
    {
        _meshRenderer = GetComponent<MeshRenderer>();
        _meshFilter = GetComponent<MeshFilter>();

        if (_meshFilter.sharedMesh == null) {
            _meshFilter.sharedMesh = mesh;
        }

        base.Awake();
    }

#if UNITY_EDITOR
    void Reset()
    {
        mesh = AssetDatabase.LoadAssetAtPath<Mesh>("Assets/Procedural/RTRect_Mesh.asset");
    }

    void OnValidate()
    {
        if (mesh == null) {
            mesh = AssetDatabase.LoadAssetAtPath<Mesh>("Assets/Procedural/RTRect_Mesh.asset");
        }
    }
#endif

    protected override void Start()
    {
#if UNITY_EDITOR
        if (!EditorApplication.isPlaying) {
            return;
        }
#endif

        base.Start();
    }

    protected override void Update()
    {
#if UNITY_EDITOR
        if(!EditorApplication.isPlaying) {
            return;
        }
#endif
        base.Update();
    }
}
