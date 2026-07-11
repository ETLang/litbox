using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace Litbox.Portfolio {
    public class LitboxDemoSceneExporter
    {
        [System.Serializable]
        struct ObjectJson
        {
            public bool active;
            public int id;
            public string name;
            public int parentId;
            public Vector2 position;
            public float depth;
            public float rotation;
            public Vector2 scale;
        }

        [System.Serializable]
        struct CameraJson
        {
            public int ownerId;
            public float verticalSize;
            public float exposure;
        }

        [System.Serializable]
        struct SimulationJson
        {
            public int ownerId;
            public int width;
            public int height;
            public int raysPerFrame;
            public float integrationInterval;
            public int photonBounces;
        }

        [System.Serializable]
        struct RaytracedJson
        {
            public int ownerId;
            public int sortOrder;
            public float logDensity;
            public float roughness;
            public float heightScale;
            public Color albedo;
            public string albedoMap;
            public string logDensityMap;
            public string sdfNormalMap;
            public string primitiveShape;
        }

        [System.Serializable]
        struct SpriteJson
        {
            public int ownerId;
            public int layer; // negative = renders before additive simulation. Positive = renders after
            public int sortOrder;
            public float opacity;
            public string image;
            public Color colorMod; // albedo color is effectively image * colorMod
            public Color ambient;
            public Color emissive;
            public Color simContribution;
            public float simBlur;
            public string primitiveShape;
        }

        [System.Serializable]
        struct PointLightJson
        {
            public int ownerId;
            public Color color;
            public float intensity;
            public int bounces;
        }

        [System.Serializable]
        struct SpotlightJson
        {
            public int ownerId;
            public Color color;
            public float intensity;
            public float pinch;
            public int bounces;
        }

        [System.Serializable]
        struct LaserLightJson
        {
            public int ownerId;
            public Color color;
            public float intensity;
            public int bounces;
        }

        [System.Serializable]
        struct DirectionalLightJson
        {
            public int ownerId;
            public Color color;
            public float intensity;
            public int bounces;
        }

        [System.Serializable]
        struct AmbientLightJson
        {
            public int ownerId;
            public Color color;
            public float intensity;
            public int bounces;
        }

        [System.Serializable]
        public struct TextureAtlasKeyJson
        {
            public string textureName;
            public string atlasName;
            public string uvTransform;
        }

        [System.Serializable]
        struct SceneDescription
        {
            public SimulationJson[] simulations;
            public ObjectJson[] objects;
            public CameraJson[] cameras;
            public RaytracedJson[] raytraced;
            public SpriteJson[] sprites;
            public PointLightJson[] pointLights;
            public SpotlightJson[] spotlights;
            public LaserLightJson[] laserLights;
            public DirectionalLightJson[] directionalLights;
            public AmbientLightJson[] ambientLights;
            public TextureAtlasKeyJson[] textureAtlasKeys;
        }

        private static T NullOr<T,S>(S src, System.Func<S,T> or) where S : UnityEngine.Object where T : class
        {
            if(!src) { return null; }
            return or(src);
        }

        private static ObjectJson ToJsonStruct(GameObject obj)
        {
            var json = new ObjectJson()
            {
                active = obj.activeSelf,
                id = obj.GetEntityId(),
                name = obj.name,
                parentId = obj.transform.parent?.gameObject.GetEntityId() ?? -1,
                position = obj.transform.localPosition,
                depth = obj.transform.localPosition.z,
                rotation = obj.transform.localEulerAngles.z,
                scale = obj.transform.localScale,
            };

            return json;
        }

        private static CameraJson ToJsonStruct(Camera cam)
        {
            var toneMapper = cam.GetComponent<ForceHDR_UE5>();

            var json = new CameraJson()
            {
                ownerId = cam.gameObject.GetEntityId(),
                verticalSize = cam.orthographicSize,
                exposure = toneMapper?.exposure ?? 0.0f
            };

            return json;
        }

        private static RaytracedJson ToJsonStruct(RTObject obj)
        {
            var json = new RaytracedJson()
            {
                ownerId = obj.gameObject.GetEntityId(),
                sortOrder = obj.sortingOrder,
                logDensity = obj.substrateLogDensity,
                roughness = 1 - obj.particleAlignment,
                heightScale = obj.heightScale,
                albedo = obj.color,
                albedoMap = NullOr(obj.texture, t => t.name),
                logDensityMap = null,
                sdfNormalMap = NullOr(obj.normal, n => n.name),
                primitiveShape = obj is RTRect ? "rect" : obj is RTEllipse ? "ellipse" : null
            };

            return json;
        }

        private static SpriteJson ToJsonStruct(RTDemoSprite sprite)
        {
            var json = new SpriteJson()
            {
                ownerId = sprite.gameObject.GetEntityId(),
                layer = (int)sprite.layer,
                sortOrder = sprite.sortingOrder,
                opacity = sprite.colorMod.a,
                image = NullOr(sprite.texture, t => t.name),
                colorMod = sprite.colorMod,
                ambient = sprite.ambience,
                emissive = sprite.emissive,
                simContribution = sprite.lightMod,
                simBlur = sprite.lightDetail
            };

            return json;
        }

        private static SimulationJson ToJsonStruct(Simulation sim)
        {
            var json = new SimulationJson()
            {
                ownerId = sim.gameObject.GetEntityId(),
                width = sim.width,
                height = sim.height,
                raysPerFrame = sim.raysPerFrame,
                integrationInterval = sim.integrationInterval,
                photonBounces = sim.photonBounces
            };

            return json;
        }

        private static PointLightJson ToJsonStruct(RTPointLight light)
        {
            var json = new PointLightJson()
            {
                ownerId = light.gameObject.GetEntityId(),
                color = light.GetComponent<SpriteRenderer>().color,
                intensity = light.intensity * light.intensity,
                bounces = (int)light.bounces
            };

            return json;
        }


        private static SpotlightJson ToJsonStruct(RTSpotLight light)
        {
            var json = new SpotlightJson()
            {
                ownerId = light.gameObject.GetEntityId(),
                color = light.GetComponent<SpriteRenderer>().color,
                intensity = light.intensity * light.intensity,
                pinch = light.pinch,
                bounces = (int)light.bounces
            };

            return json;
        }

        private static LaserLightJson ToJsonStruct(RTLaserLight light)
        {
            var json = new LaserLightJson()
            {
                ownerId = light.gameObject.GetEntityId(),
                color = light.GetComponent<SpriteRenderer>().color,
                intensity = light.intensity * light.intensity,
                bounces = (int)light.bounces
            };

            return json;
        }

        private static DirectionalLightJson ToJsonStruct(RTDirectionalLight light)
        {
            var json = new DirectionalLightJson()
            {
                ownerId = light.gameObject.GetEntityId(),
                color = light.GetComponent<SpriteRenderer>().color,
                intensity = light.intensity * light.intensity,
                bounces = (int)light.bounces
            };

            return json;
        }

        private static AmbientLightJson ToJsonStruct(RTAmbientLight light)
        {
            var json = new AmbientLightJson()
            {
                ownerId = light.gameObject.GetEntityId(),
                color = light.GetComponent<SpriteRenderer>().color,
                intensity = light.intensity * light.intensity,
                bounces = (int)light.bounces
            };

            return json;
        }


        private static IEnumerable<GameObject> GetAncestry(GameObject obj)
        {
            if(obj == null) { return new List<GameObject>(); }
            var list = new List<GameObject>();
            list.Add(obj);
            list.AddRange(GetAncestry(obj.transform.parent?.gameObject));
            return list;
        }

        [MenuItem("Litbox/Export Scene to JSON")]
        public static void ExportSceneToJson()
        {
            string path = EditorUtility.SaveFilePanel(
                "Export Scene to JSON",
                "Assets",
                "scene_export.json",
                "json");

            if (!string.IsNullOrEmpty(path))
            {
                var gameObjects = Object.FindObjectsByType<GameObject>(FindObjectsInactive.Include, FindObjectsSortMode.None);
                var cameras = Object.FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
                var rtObjects = Object.FindObjectsByType<RTObject>(FindObjectsInactive.Include, FindObjectsSortMode.None);
                var rtLightSources = Object.FindObjectsByType<RTLightSource>(FindObjectsInactive.Include, FindObjectsSortMode.None);
                var rtSprites = Object.FindObjectsByType<RTDemoSprite>(FindObjectsInactive.Include, FindObjectsSortMode.None);
                var simulations = Object.FindObjectsByType<Simulation>(FindObjectsInactive.Include, FindObjectsSortMode.None);

                var output = new SceneDescription();

                output.simulations = simulations
                    .Select(ToJsonStruct)
                    .ToArray();

                output.objects = gameObjects
                    .Select(ToJsonStruct)
                    .ToArray();

                output.cameras = cameras
                    .Select(ToJsonStruct)
                    .ToArray();
                
                output.raytraced = rtObjects
                    .Select(ToJsonStruct)
                    .ToArray();

                output.pointLights = rtLightSources
                    .OfType<RTPointLight>()
                    .Select(ToJsonStruct)
                    .ToArray();

                output.spotlights = rtLightSources
                    .OfType<RTSpotLight>()
                    .Select(ToJsonStruct)
                    .ToArray();

                output.laserLights = rtLightSources
                    .OfType<RTLaserLight>()
                    .Select(ToJsonStruct)
                    .ToArray();

                output.directionalLights = rtLightSources
                    .OfType<RTDirectionalLight>()
                    .Select(ToJsonStruct)
                    .ToArray();

                output.ambientLights = rtLightSources
                    .OfType<RTAmbientLight>()
                    .Select(ToJsonStruct)
                    .ToArray();

                output.sprites = rtSprites
                    .Select(ToJsonStruct)
                    .ToArray();

                var allTextures = new Dictionary<string, Texture>();
                void AddTexture(Texture t)
                {
                    if (t != null && !allTextures.ContainsKey(t.name)) allTextures[t.name] = t;
                }
                foreach (var rtObject in rtObjects)
                {
                    AddTexture(rtObject.texture);
                    AddTexture(rtObject.normal);
                }
                foreach (var sprite in rtSprites)
                {
                    AddTexture(sprite.texture);
                }

                string jsonDir = Path.GetDirectoryName(path);
                string atlasRelDir = Path.GetFileNameWithoutExtension(path) + "_atlases";
                output.textureAtlasKeys = TextureAtlasBaker.BakeAtlases(allTextures, Path.Combine(jsonDir, atlasRelDir), atlasRelDir);

                var sceneData = JsonUtility.ToJson(output, true);
                File.WriteAllText(path, sceneData);
                Debug.Log($"Scene exported to: {path}");
            }
        }
    }
}