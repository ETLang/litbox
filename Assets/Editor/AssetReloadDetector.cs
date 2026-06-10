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

        LitboxComponent.CheckForGlobalChanges();
    }
}
