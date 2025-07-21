using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Random=System.Random;

[Serializable]
public struct JsonSceneData {
    public Color ambientLightColor;
    public float ambientLightIntensity;
    public Color backgroundColor;
    public float backgroundDensity;
    public uint[] substrateSeeds;
    public uint[] substrateSeedsV2;
    public JsonLightData[] lights;
}

[Serializable]
public struct JsonLightData {
    public string type;
    public Vector2 position;
    public float angle;
    public Vector2 scale;
    public Color color;
    public float intensity;
}


public class TrainingManager : MonoBehaviour {
    [SerializeField] bool _train;
    [SerializeField] bool _continuePreviousSession;
    [SerializeField] public string outputFolder;
    [SerializeField] int _samplesToGenerate = 5000;
    [SerializeField] SimulationProfile[] inputProfiles;
    [SerializeField] SimulationProfile convergenceProfile;

    [SerializeField] Simulation _simulation;
    [SerializeField] TrainingSubstrate _substrateA;
    [SerializeField] TrainingSubstrate _substrateB;
    [SerializeField] TrainingSubstrate _substrateC;
    [SerializeField] RTObject _backgroundSubstrate;
    [SerializeField] RTAmbientLight _ambientLight;
    [SerializeField] GameObject _lightContainer;

    [Header("Prefabs")]
    [SerializeField] GameObject _directionalLightPrefab;
    [SerializeField] GameObject _pointLightPrefab;
    [SerializeField] GameObject _spotLightPrefab;
    [SerializeField] GameObject _laserLightPrefab;

    [Header("Utility")]
    [SerializeField] Renderer _sampleDisplay;
    [SerializeField] int _showProfile;
    [SerializeField] string _debugScenePath;


    Random _rand;
    List<GameObject> _generatedObjects = new List<GameObject>();
    int _samplesToGenerateThisSession;
    int _generatedSamples;
    int _sampleIdOffset = 0;
    string _datasetName;
    string _datasetPath;

    int _activeProfile;
    string _imagePathPNG;
    string _imagePathEXR;
    RenderTexture[] profileSamples;

    public bool IsGenerating { get; private set; }

    int FindNextAvailableSampleID()
    {
        int i = 0;
        while (File.Exists(Path.Combine(_datasetPath, $"Input_0_{i:0000}.png")))
            i++;
        return i;
    }

    void AdvanceTrainingState() {
        while(_generatedSamples < _samplesToGenerateThisSession) {
            int sampleId;
            if(_activeProfile == inputProfiles.Length) {
                _imagePathPNG = null;
                _imagePathEXR = null;
                _generatedSamples++;
                if(profileSamples != null) {
                    foreach(var sample in profileSamples) {
                        DestroyImmediate(sample);
                    }
                }

                if (_generatedSamples == _samplesToGenerateThisSession)
                    break;

                sampleId = _generatedSamples + _sampleIdOffset;

                JsonSceneData sceneDesc;
                var scenePath = Path.Combine(_datasetPath, $"Scene_{sampleId:0000}.json");

                if (!File.Exists(scenePath)) {
                    sceneDesc = GenerateRandomSceneDescription();
                    File.WriteAllText(scenePath, JsonUtility.ToJson(sceneDesc, true));
                } else {
                    sceneDesc = JsonUtility.FromJson<JsonSceneData>(File.ReadAllText(scenePath));
                }

                LoadSceneFromDescription(sceneDesc);

                _activeProfile = -1;
            } else {
                sampleId = _generatedSamples + _sampleIdOffset;
            }

            _activeProfile++;

            if(_activeProfile == inputProfiles.Length) {
                _imagePathPNG = Path.Combine(_datasetPath, $"Output_{sampleId:0000}.png");
                _imagePathEXR = Path.Combine(_datasetPath, $"Output_{sampleId:0000}.exr");
            } else {
                _imagePathPNG = Path.Combine(_datasetPath, $"Input_{_activeProfile}_{sampleId:0000}.png");
                _imagePathEXR = Path.Combine(_datasetPath, $"Input_{_activeProfile}_{sampleId:0000}.exr");
            }

            if(File.Exists(_imagePathPNG) && File.Exists(_imagePathEXR)) {
                continue;
            }

            var profile = (_activeProfile == inputProfiles.Length) ? convergenceProfile : inputProfiles[_activeProfile];
            _simulation.LoadProfile(profile);
            return;
        }

        IsGenerating = false;
    }

    void Start() {
        _rand = new Random(Environment.TickCount);

        if(_continuePreviousSession) {
            _datasetPath = Directory.GetDirectories(outputFolder).OrderByDescending(s => s).FirstOrDefault();

            if(string.IsNullOrEmpty(_datasetPath)) {
                throw new Exception("No previous session to update!");
            }

            _datasetName = Path.GetDirectoryName(_datasetPath);
        } else {
            _datasetName = DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss");
            _datasetPath = Path.Combine(outputFolder, _datasetName);
            Directory.CreateDirectory(_datasetPath);
        }

        if(!string.IsNullOrEmpty(_debugScenePath)) {
            LoadSceneFromDescription(File.ReadAllText(_debugScenePath));
            _train = false;
        } else if(_train) {
            Debug.Log($"Beginning training session {_datasetName}");

            BeginComposingTrainingImages(false);
        }

        _simulation.OnStep += OnSimulationStep;
        _simulation.OnConverged += OnSimulationConverged;


        /*

        Important Sampling thoughts:
        - Idea is to use nonuniform PDF to fire more photons in directions where they matter more.
        - Conceivably, you have 1 PDF per random variable
        - Point light has 3 random variables (2 for position, 1 for direction)
        - When a photon is fired, measure whether it escapes.
        - If the photon did not escape, tally it on the PDF
        - After N samples are collected, compile the PDF into a quick integral LUT
        - Use the integral LUT to distribute future rays.
        - Scale photon intensity inversely with photon density for conservation of energy.
        - Since photon energy is an integer, consider dithering photon density.

        New Light Types:
        - Hemispherical Ambient Light
        */

    }

    void OnDisable() {
        _simulation.OnStep -= OnSimulationStep;
        _simulation.OnConverged -= OnSimulationConverged;

        if(profileSamples != null) {
            foreach(var sample in profileSamples) {
                DestroyImmediate(sample);
            }
            profileSamples = null;
        }
    }

    void OnSimulationStep(int frameCount) {
        if(IsGenerating) {
            if(_sampleDisplay != null && profileSamples[_showProfile] != null) {
                _sampleDisplay.material.SetTexture("_MainTex", profileSamples[_showProfile]);
            }
        }
    }

    void OnSimulationConverged() {
        if(IsGenerating) {
            _simulation.SimulationOutputHDR.SaveTextureEXR(_imagePathEXR);
            _simulation.SimulationOutputToneMapped.SaveTexturePNG(_imagePathPNG);
            if(_activeProfile == inputProfiles.Length) {
                _simulation.SimulationOutputToneMapped.SaveTexturePNG(_imagePathPNG);
                Debug.Log($"Completed Scene {_generatedSamples:0000}");
            } else {
                var sample = new RenderTexture(_simulation.SimulationOutputToneMapped);
                var tmp = RenderTexture.active;
                Graphics.Blit(_simulation.SimulationOutputToneMapped, sample);
                RenderTexture.active = tmp;
                profileSamples[_activeProfile] = sample;
            }

            AdvanceTrainingState();
        }
    }

    public JsonSceneData GenerateRandomSceneDescription() {
        var output = new JsonSceneData();

        output.ambientLightColor = _rand.NextLightColor();
        output.ambientLightIntensity = _rand.NextRange(0,0.5f,-0.5f);
        output.backgroundColor = Color.white; // _rand.NextLightColor();
        output.backgroundDensity = _rand.NextRange(-4, -2, -0.4f);
        output.lights = new JsonLightData[_rand.Next(3) + 1];

        for(int i = 0;i < output.lights.Length;i++) {
            var light = new JsonLightData();

            light.type = _rand.NextWeightedOption(new Dictionary<string,float> {
                { "Directional", 0.0f }, // Disabled because it has bugs
                { "Point", 0.25f },
                { "Spot", 0.25f },
                { "Laser", 0.1f }
            });

            light.color = _rand.NextLightColor();
            light.intensity = _rand.NextRange(0.01f, 1.5f, 0.2f);

            switch(light.type) {
            case "Directional":
                light.position = Vector2.zero;
                light.angle = _rand.NextRange(0,360);
                light.scale = new Vector2(1, 1);
                break;
            case "Point":
                light.position = new Vector2(_rand.NextRange(-5, 5), _rand.NextRange(-5, 5));
                light.angle = 0;
                var size = _rand.NextRange(0.4f, 5, 0.1f);
                light.scale = new Vector2(size, size);
                break;
            case "Spot":
                light.position = new Vector2(_rand.NextRange(-7, 7), _rand.NextRange(-7, 7));
                var baseAngle = Mathf.Acos(light.position.x / light.position.magnitude) * 180 / Mathf.PI;
                if(light.position.y < 0)
                    baseAngle *= -1;
                baseAngle += 270;
                light.angle = baseAngle + _rand.NextRange(-80, 80);
                light.scale = new Vector2(_rand.NextRange(0.3f,1.5f,0.3f), _rand.NextRange(0.5f, 3.0f));
                break;
            case "Laser":
                light.position = new Vector2(_rand.NextRange(-3, 3), _rand.NextRange(-3, 3));
                light.angle = _rand.NextRange(0,360);
                light.scale = new Vector2(_rand.NextRange(0.01f, 0.2f, 0.1f), 1);
                break;
            default:
                Debug.LogError("What light type is this? " + light.type);
                break;
            }

            output.lights[i] = light;
        }

        List<uint> substrates = new List<uint>();

        substrates.Add((uint)_rand.Next());
        if(_rand.NextBool()) {
            substrates.Add((uint)_rand.Next());
            if(_rand.NextBool()) {
                substrates.Add((uint)_rand.Next());
            }
        }

        output.substrateSeedsV2 = substrates.ToArray();

        return output;
    }

    public void LoadSceneFromDescription(string json) {
        LoadSceneFromDescription(JsonUtility.FromJson<JsonSceneData>(json));
    }

    public void LoadSceneFromDescription(JsonSceneData data) {
        foreach(var oldObject in _generatedObjects)
            DestroyImmediate(oldObject);
        _generatedObjects.Clear();

        _ambientLight.GetComponent<SpriteRenderer>().color = data.ambientLightColor;
        _ambientLight.intensity = data.ambientLightIntensity;
        _backgroundSubstrate.substrateLogDensity = data.backgroundDensity;
        _backgroundSubstrate.GetComponent<SpriteRenderer>().color = data.backgroundColor;

        for(int i = 0;i < data.lights.Length;i++) {
            var lightData = data.lights[i];

            GameObject newLightGO = null;
            switch(lightData.type) {
            case "Directional":
                newLightGO = Instantiate(_directionalLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Directional Light";
                break;
            case "Point":
                newLightGO = Instantiate(_pointLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Point Light";
                break;
            case "Spot":
                newLightGO = Instantiate(_spotLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Spot Light";
                break;
            case "Laser":
                newLightGO = Instantiate(_laserLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Laser Light";
                break;
            default:
                Debug.LogError("What light type is this? " + lightData.type);
                break;
            }

            newLightGO.transform.localPosition = lightData.position;
            newLightGO.transform.localRotation = Quaternion.Euler(0,0,lightData.angle);
            newLightGO.transform.localScale = new Vector3(lightData.scale.x, lightData.scale.y, 1);

            _generatedObjects.Add(newLightGO);

            RTLightSource newLight = newLightGO.GetComponent<RTLightSource>();
            SpriteRenderer lightSprite = newLightGO.GetComponent<SpriteRenderer>();

            lightSprite.color = lightData.color;
            newLight.bounces = 10; // We want a high fidelity result!
            newLight.intensity = lightData.intensity;
        }

        uint[] seeds = null;
        int version = 1;

        if(data.substrateSeedsV2 != null && data.substrateSeedsV2.Length != 0) {
            seeds = data.substrateSeedsV2;
            version = 2;
        } else if(data.substrateSeeds != null && data.substrateSeeds.Length != 0) {
            seeds = data.substrateSeeds;
            version = 1;
        }

        _substrateB.gameObject.SetActive(seeds.Length >= 2);
        _substrateC.gameObject.SetActive(seeds.Length >= 3);

        _substrateA.GenerateRandom(seeds[0], version);
        _substrateA.ValidateAndApply();
        if(_substrateB.gameObject.activeSelf) {
            _substrateB.GenerateRandom(seeds[1], version);
            _substrateB.ValidateAndApply();
        }
        if(_substrateC.gameObject.activeSelf) {
            _substrateC.GenerateRandom(seeds[2], version);
            _substrateC.ValidateAndApply();
        }
    }

    public void SetupRandomScene() {
        foreach(var oldObject in _generatedObjects)
            DestroyImmediate(oldObject);
        _generatedObjects.Clear();

        _ambientLight.GetComponent<SpriteRenderer>().color = _rand.NextLightColor();
        _ambientLight.intensity = _rand.NextRange(0,0.5f,-0.5f);
        _backgroundSubstrate.substrateLogDensity = _rand.NextRange(-4, -2, -0.4f);
        _backgroundSubstrate.GetComponent<SpriteRenderer>().color = _rand.NextLightColor();

        int lightCount = _rand.Next(3) + 1;

        for(int i = 0;i < lightCount;i++) {
            var lightType = _rand.NextWeightedOption(new Dictionary<string,float> {
                { "Directional", 0.0f }, // Disabled because it has bugs
                { "Point", 0.25f },
                { "Spot", 0.25f },
                { "Laser", 0.1f }
            });

            GameObject newLightGO = null;
            switch(lightType) {
            case "Directional":
                newLightGO = Instantiate(_directionalLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Directional Light";
                newLightGO.transform.localRotation = Quaternion.Euler(0,0,_rand.NextRange(0,360));
                break;
            case "Point":
                newLightGO = Instantiate(_pointLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Point Light";
                newLightGO.transform.localPosition = new Vector3(_rand.NextRange(-5, 5), _rand.NextRange(-5, 5), 0);
                var size = _rand.NextRange(0.4f, 5, 0.1f);
                newLightGO.transform.localScale = new Vector3(size, size, 1);
                break;
            case "Spot":
                newLightGO = Instantiate(_spotLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Spot Light";
                var pos = new Vector3(_rand.NextRange(-7, 7), _rand.NextRange(-7, 7), 0);
                newLightGO.transform.localPosition = pos;
                var baseAngle = Mathf.Acos(pos.x / pos.magnitude) * 180 / Mathf.PI;
                if(pos.y < 0)
                    baseAngle *= -1;
                baseAngle += 270;
                newLightGO.transform.localRotation = Quaternion.Euler(0,0,baseAngle + _rand.NextRange(-80, 80));
                newLightGO.transform.localScale = new Vector3(_rand.NextRange(0.3f,1.5f,0.3f), _rand.NextRange(0.5f, 3.0f), 1);
                break;
            case "Laser":
                newLightGO = Instantiate(_laserLightPrefab, _lightContainer.transform);
                newLightGO.name = "Random Laser Light";
                newLightGO.transform.localPosition = new Vector3(_rand.NextRange(-3, 3), _rand.NextRange(-3, 3), 0);
                newLightGO.transform.localRotation = Quaternion.Euler(0,0,_rand.NextRange(0,360));
                break;
            default:
                Debug.LogError("What light type is this? " + lightType);
                break;
            }

            _generatedObjects.Add(newLightGO);

            RTLightSource newLight = newLightGO.GetComponent<RTLightSource>();
            SpriteRenderer lightSprite = newLightGO.GetComponent<SpriteRenderer>();

            lightSprite.color = _rand.NextLightColor();
            newLight.bounces = 10; // We want a high fidelity result!
            newLight.intensity = _rand.NextRange(0.1f, 5.0f, 0.2f);
        }

        _substrateB.gameObject.SetActive(_rand.NextBool());
        _substrateC.gameObject.SetActive(_substrateB.gameObject.activeSelf && _rand.NextBool());

        _substrateA.GenerateRandom((uint)_rand.Next());
        _substrateA.ValidateAndApply();
        if(_substrateB.gameObject.activeSelf) {
            _substrateB.GenerateRandom((uint)_rand.Next());
            _substrateB.ValidateAndApply();
        }
        if(_substrateC.gameObject.activeSelf) {
            _substrateC.GenerateRandom((uint)_rand.Next());
            _substrateC.ValidateAndApply();
        }

/*

        Training Data Production Procedure:

        - Ambient Intensity: 0-0.5
        - Background Density: -4 - -2

        Single-Light Process:
        1. Randomly create a single light source
        2. Randomly configure background substrate, and two layers of random substrates
        2. Set up simulation to run N photons with like 10 bounces.
        3. Run 1 frame
        4. Capture output as source for training.
        5. Ensure simulation occurs < 3 ms
        6. Configure simulation for optimal speed
        7. Run until convergence
        8. Capture output as reference for training.
        9. Save light configuration and substrate configurations to JSON file for future debugging.
        10. Repeat at least 1000 times
    
        Multi-Light Process:
        - Same, except use 1-4 randomly configured light sources

*/
    }

    public void AcceptCurrentGeneration()
    {
        OnSimulationConverged();
    }

    public void BeginComposingTrainingImages(bool composeOnlyCurrentScene)
    {
        if (_continuePreviousSession) {
            var previousProfiles = new List<SimulationProfile>();
            int i = 0;
            while (true) {
                var profilePath = Path.Combine(_datasetPath, $"inputProfile_{i}.json");
                if (!File.Exists(profilePath)) {
                    break;
                }

                var oldProfile = JsonUtility.FromJson<SimulationProfile>(File.ReadAllText(profilePath));
                previousProfiles.Add(oldProfile);
                i++;
            }

            var inputProfileList = inputProfiles.ToList();
            i = 0;
            while (i < inputProfileList.Count) {
                if (previousProfiles.Contains(inputProfileList[i])) {
                    inputProfileList.RemoveAt(i);
                } else {
                    i++;
                }
            }

            for (i = 0; i < inputProfileList.Count; i++) {
                previousProfiles.Add(inputProfileList[i]);
            }

            inputProfiles = previousProfiles.ToArray();
        }

        for (int i = 0; i < inputProfiles.Length; i++) {
            var path = Path.Combine(_datasetPath, $"inputProfile_{i}.json");
            if (!File.Exists(path)) {
                File.WriteAllText(path, JsonUtility.ToJson(inputProfiles[i], true));
            }
        }
        profileSamples = new RenderTexture[inputProfiles.Length];

        if (composeOnlyCurrentScene) {
            _sampleIdOffset = FindNextAvailableSampleID();
            _samplesToGenerateThisSession = 1;
            _activeProfile = -1;
            _generatedSamples = 0;
        } else {
            _sampleIdOffset = 0;
            _samplesToGenerateThisSession = _samplesToGenerate;
            _activeProfile = inputProfiles.Length;
            _generatedSamples = -1;
        }
        IsGenerating = true;
        AdvanceTrainingState();
    }
}