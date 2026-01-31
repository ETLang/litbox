using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteAlways]
public class RTEllipse : RTObject
{
    MeshRenderer _meshRenderer;
    MeshFilter _meshFilter;
    Mesh _mesh;
    Material _rtMat;

    static Mesh _sharedMesh;
    static Mesh GetSharedMesh()
    {
        if (_sharedMesh == null) {
            const int segmentCount = 32;
            var vertices = new Vector3[segmentCount + 1];
            var normals = new Vector3[segmentCount + 1];
            var uvs = new Vector2[segmentCount + 1];
            var indices = new int[segmentCount * 3];

            vertices[0] = Vector3.zero;
            normals[0] = new Vector3(0,0,-1);
            uvs[0] = new Vector2(0.5f, 0.5f);

            for (int i = 0; i < segmentCount; i++) {
                float angle = (float)i / segmentCount * Mathf.PI * 2f;
                float x = Mathf.Cos(angle) * 0.5f;
                float y = Mathf.Sin(angle) * 0.5f;
                vertices[i + 1] = new Vector3(x, y, 0f);
                normals[i + 1] = new Vector3(x, y, 0f);
                uvs[i + 1] = new Vector2(x + 0.5f, y + 0.5f);

                indices[i * 3 + 0] = 0;
                indices[i * 3 + 2] = i + 1;
                indices[i * 3 + 1] = (i + 1) % segmentCount + 1;
            }

            _sharedMesh = new Mesh {
                vertices = vertices,
                normals = normals,
                uv = uvs,
                triangles = indices,
                name = "RTEllipse mesh",
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
