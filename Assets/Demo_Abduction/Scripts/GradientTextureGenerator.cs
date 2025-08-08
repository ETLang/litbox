using UnityEngine;

public class GradientTextureGenerator : MonoBehaviour
{
    public Gradient gradient; // Assign this in the Inspector
    public Texture2D texture;
    private Material mat;

    void Start()
    {
        mat = GetComponent<Renderer>().sharedMaterial;
        GenerateGradientTexture();
    }

    void GenerateGradientTexture()
    {
        texture = new Texture2D(1, 2048);
        texture.wrapMode = TextureWrapMode.Clamp;
        for (int i = 0; i < texture.height; i++) {
            // Sample the gradient at a normalized position (0 to 1)
            Color color = gradient.Evaluate((float)i / (texture.height - 1));
            texture.SetPixel(0, i, color);
        }
        texture.Apply();

        // Pass the texture to the shader
        mat.SetTexture("_MainTex", texture);
    }
}