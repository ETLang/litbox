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

    static Mesh _sharedMesh;
    static Mesh GetSharedMesh()
    {
        if (_sharedMesh == null) {
            var vertices = new Vector3[] {
                new Vector3(-0.5f,  0.5f,  0),
                new Vector3(    0,     0,  0),
                new Vector3(-0.5f, -0.5f, 0),
                new Vector3( 0.5f, 0.5f, 0),
                new Vector3(    0,    0, 0),
                new Vector3(-0.5f, 0.5f, 0),
                new Vector3( 0.5f,-0.5f, 0),
                new Vector3(    0,    0, 0),
                new Vector3( 0.5f, 0.5f, 0),
                new Vector3(-0.5f,-0.5f, 0),
                new Vector3(    0,    0, 0),
                new Vector3( 0.5f,-0.5f, 0),
            };
            var normals = new Vector3[] {
                new Vector3(-1, 0, 0),
                new Vector3(-1, 0, 0),
                new Vector3(-1, 0, 0),
                new Vector3( 0, 1, 0),
                new Vector3( 0, 1, 0),
                new Vector3( 0, 1, 0),
                new Vector3( 1, 0, 0),
                new Vector3( 1, 0, 0),
                new Vector3( 1, 0, 0),
                new Vector3( 0, -1, 0),
                new Vector3( 0, -1, 0),
                new Vector3( 0, -1, 0),
            };
            var uvs = new Vector2[] {
                new Vector2(0, 1),
                new Vector2(0.5f, 0.5f),
                new Vector2(0, 0),
                new Vector2(1, 1),
                new Vector2(0.5f, 0.5f),
                new Vector2(0, 1),
                new Vector2(1, 0),
                new Vector2(0.5f, 0.5f),
                new Vector2(1, 1),
                new Vector2(0, 0),
                new Vector2(0.5f, 0.5f),
                new Vector2(1, 0),
            };
            var indices = new int[] {
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
                9, 10, 11,
            };
            _sharedMesh = new Mesh {
                vertices = vertices,
                normals = normals,
                uv = uvs,
                triangles = indices,
                name = "RTRect mesh",
            };
            _sharedMesh.UploadMeshData(true);
        }
        return _sharedMesh;
    }

    protected override void Awake()
    {
        _meshRenderer = GetComponent<MeshRenderer>();
        _meshFilter = GetComponent<MeshFilter>();

        if (_meshFilter.sharedMesh == null) {
            _meshFilter.sharedMesh = GetSharedMesh();
        }

        base.Awake();
    }

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
