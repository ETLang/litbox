using UnityEngine;
using UnityEditor;

public class AssetReloader : MonoBehaviour
{
    // This method will be called when you click on a menu item in the Unity editor
    [MenuItem("Tools/Force Reload Assets")]
    public static void ForceReloadAssets()
    {
        AssetDatabase.Refresh();
        Debug.Log("Assets reloaded!");
    }
}