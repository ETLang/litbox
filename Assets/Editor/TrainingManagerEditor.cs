using System.IO;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEditor;
using System.Linq;
using System.Text.RegularExpressions;

[CustomEditor(typeof(TrainingManager))]
public class TrainingManagerDrawer : Editor
{
    public override void OnInspectorGUI()
    {
        TrainingManager manager = (TrainingManager)target;

        DrawDefaultInspector();

        if (manager.IsGenerating) {
            if (GUILayout.Button("Accept Current Scene")) {
                manager.AcceptCurrentGeneration();
            }
        } else {
            if (GUILayout.Button("Compose Current Scene")) {
                manager.BeginComposingTrainingImages(true);
            }
        }

        if (GUILayout.Button("Generate Random Scene")) {
            manager.SetupRandomScene();
        }

        if(GUILayout.Button("Choose Output Folder")) {
            var path = EditorUtility.SaveFolderPanel("Choose Output Folder", manager.outputFolder, "training_output");

            if(!string.IsNullOrEmpty(path)) {
                manager.outputFolder = path;
            }
        }

        if(GUILayout.Button("Consolidate Latest Into Previous")) {
            var previousSessions = Directory.EnumerateDirectories(manager.outputFolder).OrderByDescending(path => path).ToArray();

            if(previousSessions.Length < 2) {
                Debug.Log("Not enough sessions to consolidate");
                return;
            }

            var lastSession = previousSessions[0];
            var targetSession = previousSessions[1];

            var targetFiles = Directory.EnumerateFiles(targetSession).OrderByDescending(f => f).ToArray();
            var lastFile = Path.GetFileNameWithoutExtension(targetFiles[0]);

            var pattern = new Regex("(?<=Input_|Output_)[0-9]+");

            var baseIndex = int.Parse(pattern.Match(lastFile).Value);

            foreach(var fileToMove in Directory.EnumerateFiles(lastSession)) {
                var srcName = Path.GetFileName(fileToMove);
                var srcIndex = int.Parse(pattern.Match(srcName).Value);
                var destName = Path.Combine(targetSession, pattern.Replace(srcName, (srcIndex + baseIndex).ToString("0000")));
                File.Copy(fileToMove, destName);
            }
        }
   }
}