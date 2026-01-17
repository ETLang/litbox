using System;
using UnityEditor;
using UnityEngine;

// This class is an editor script and will be automatically
// invoked by Unity whenever asset events occur.
public class AssetReloadDetector : AssetPostprocessor
{
    public static Action Reloaded;

    // Called when assets are imported, deleted, or moved.
    private static void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        Reloaded?.Invoke();

        PhotonerComponent.CheckForGlobalChanges();

        //// Log all imported assets to the console.
        //foreach (string assetPath in importedAssets) {
        //    Debug.Log($"Imported Asset: {assetPath}");
        //}

        //// Log all deleted assets.
        //foreach (string assetPath in deletedAssets) {
        //    Debug.Log($"Deleted Asset: {assetPath}");
        //}

        //// Log all moved assets, showing their old and new paths.
        //for (int i = 0; i < movedAssets.Length; i++) {
        //    Debug.Log($"Moved Asset: from '{movedFromAssetPaths[i]}' to '{movedAssets[i]}'");
        //}
    }
}
