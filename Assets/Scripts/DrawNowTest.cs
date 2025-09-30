using UnityEngine;

public class DrawNowTest : MonoBehaviour
{
    public Material mat;
    public Mesh themesh;

    private void OnPreRender()
    {
    }

    private void OnPostRender()
    {
        if (mat == null || themesh == null) return;

        GL.Clear(true, true, Color.red);
        mat.SetPass(0);
        Graphics.DrawMeshNow(themesh, Matrix4x4.Translate(new Vector3(1, 0, 4)));
    }
}
