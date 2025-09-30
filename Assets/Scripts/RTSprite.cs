using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public class RTSprite : RTObject
{
    private Sprite _previousSprite;
    private Color _previousColor;

    public Mesh sharedMesh { get; private set; }
    public Material sharedMaterial { get; private set; }

    new protected void Start() {
        base.Start();
        
        var spriteRenderer = GetComponent<SpriteRenderer>();
        spriteRenderer.material.color = spriteRenderer.color;

        _previousSprite = spriteRenderer.sprite;
        _previousColor = spriteRenderer.color;

        GetMeshData();
    }
    
    new protected void Update()
    {
        base.Update();

        var spriteRenderer = GetComponent<SpriteRenderer>();

        Changed = Changed ||
            _previousSprite != spriteRenderer.sprite ||
            _previousColor != spriteRenderer.color;

        _previousSprite = spriteRenderer.sprite;
        _previousColor = spriteRenderer.color;

        if(Changed) {
            spriteRenderer.material.color = spriteRenderer.color;
        }
    }

    void GetMeshData()
    {
        var spriteRenderer = GetComponent<SpriteRenderer>();

        Mesh mesh = new Mesh {
            vertices = System.Array.ConvertAll(spriteRenderer.sprite.vertices, v => (Vector3)v),
            uv = spriteRenderer.sprite.uv,
            triangles = System.Array.ConvertAll(spriteRenderer.sprite.triangles, i => (int)i)
        };

        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        sharedMaterial = spriteRenderer.sharedMaterial;
        sharedMesh = mesh;
    }
}
