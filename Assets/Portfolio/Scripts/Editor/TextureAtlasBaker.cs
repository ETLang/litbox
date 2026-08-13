using System;
using System.Collections.Generic;
using System.IO;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace Litbox.Portfolio
{
    // Bakes referenced textures into per-format atlas pages and computes the
    // per-texture UV remap (source UV -> atlas UV) for the exported JSON.
    public static class TextureAtlasBaker
    {
        private const int Padding = 2; // bilinear-only sampling (no mips) needs ~1px; 2px is a safety margin.
        private const int MaxPageSize = 4096;

        // DXT1 packs at 4x4-block granularity: 1 block of padding guarantees a full
        // block-boundary gap between any two adjacent textures (never a shared block).
        private const int BlockPadding = 1; // in blocks, for the packer call
        private const int BlockPaddingPixels = 4; // 1 block, for the pixel-space extrusion ring

        private enum FormatBucket
        {
            Rgba32Srgb,
            RFloat,
            RgbaFloat,
            Dxt1
        }

        private struct SourceTexture
        {
            public FormatBucket bucket;
            public int width;
            public int height;
            public float4[,] pixels;
        }

        public static LitboxDemoSceneExporter.TextureAtlasKeyJson[] BakeAtlases(
            IReadOnlyDictionary<string, Texture> texturesByName,
            string outputDirAbsolute,
            string outputDirRelativeToJson)
        {
            var sources = new Dictionary<string, SourceTexture>();
            var itemsByBucket = new Dictionary<FormatBucket, List<PackItem>>();

            foreach (var kvp in texturesByName)
            {
                string name = kvp.Key;
                Texture tex = kvp.Value;

                var bucket = Classify(tex);
                var pixels = tex.ForceReadData();
                int w = pixels.GetLength(0);
                int h = pixels.GetLength(1);

                sources[name] = new SourceTexture { bucket = bucket, width = w, height = h, pixels = pixels };

                // Dxt1 packs at 4x4-block granularity so two different textures never
                // share a block; every other bucket packs in raw pixel units.
                int itemW = bucket == FormatBucket.Dxt1 ? CeilDiv(w, 4) : w;
                int itemH = bucket == FormatBucket.Dxt1 ? CeilDiv(h, 4) : h;

                if (!itemsByBucket.TryGetValue(bucket, out var list))
                {
                    list = new List<PackItem>();
                    itemsByBucket[bucket] = list;
                }
                list.Add(new PackItem { id = name, rawWidth = itemW, rawHeight = itemH });
            }

            var placementsByName = new Dictionary<string, Placement>();
            var pageSizes = new Dictionary<(FormatBucket, int), (int w, int h)>();
            var pageBuffers = new Dictionary<(FormatBucket, int), Color[]>();

            foreach (var bucketKvp in itemsByBucket)
            {
                var bucket = bucketKvp.Key;
                bool isBlockUnit = bucket == FormatBucket.Dxt1;
                int padding = isBlockUnit ? BlockPadding : Padding;
                int maxPageSize = isBlockUnit ? MaxPageSize / 4 : MaxPageSize;

                var placements = TextureAtlasPacker.Pack(bucketKvp.Value, padding, maxPageSize);

                foreach (var placement in placements)
                {
                    var p = placement;
                    if (isBlockUnit)
                    {
                        // Convert block units back to pixels once, here - every
                        // downstream consumer of Placement deals in pixels only.
                        p.x *= 4; p.y *= 4; p.pageWidth *= 4; p.pageHeight *= 4;
                    }

                    placementsByName[p.id] = p;
                    var key = (bucket, p.page);
                    if (!pageSizes.ContainsKey(key))
                    {
                        pageSizes[key] = (p.pageWidth, p.pageHeight);
                        pageBuffers[key] = new Color[p.pageWidth * p.pageHeight];
                    }
                }
            }

            foreach (var kvp in sources)
            {
                var src = kvp.Value;
                var placement = placementsByName[kvp.Key];
                var key = (src.bucket, placement.page);
                var (pageW, pageH) = pageSizes[key];
                var buffer = pageBuffers[key];

                bool swapped = placement.rotation == 90 || placement.rotation == 270;
                int contentW = swapped ? src.height : src.width;
                int contentH = swapped ? src.width : src.height;

                BlitRotated(src.pixels, src.width, src.height, buffer, pageW, placement.x, placement.y, placement.rotation);

                if (src.bucket == FormatBucket.Dxt1)
                {
                    int roundedW = CeilDiv(contentW, 4) * 4;
                    int roundedH = CeilDiv(contentH, 4) * 4;
                    ExtrudeBlockSlack(buffer, pageW, placement.x, placement.y, contentW, contentH, roundedW - contentW, roundedH - contentH);
                    ExtrudeEdges(buffer, pageW, pageH, placement.x, placement.y, roundedW, roundedH, BlockPaddingPixels);
                }
                else
                {
                    ExtrudeEdges(buffer, pageW, pageH, placement.x, placement.y, contentW, contentH, Padding);
                }
            }

            Directory.CreateDirectory(outputDirAbsolute);
            var atlasFileByPage = new Dictionary<(FormatBucket, int), string>();

            foreach (var pageKvp in pageBuffers)
            {
                var (bucket, page) = pageKvp.Key;
                var (w, h) = pageSizes[pageKvp.Key];

                var tex = CreatePageTexture(bucket, w, h);
                tex.SetPixels(pageKvp.Value);
                tex.Apply(false, false);

                string fileName = $"atlas_{BucketSlug(bucket)}_{page}.{BucketExtension(bucket)}";
                byte[] bytes;
                if (bucket == FormatBucket.Rgba32Srgb)
                {
                    bytes = tex.EncodeToPNG();
                }
                else if (bucket == FormatBucket.Dxt1)
                {
                    bytes = EncodeDxt1Container(tex, w, h);
                }
                else
                {
                    bytes = tex.EncodeToEXR(Texture2D.EXRFlags.CompressZIP | Texture2D.EXRFlags.OutputAsFloat);
                }
                File.WriteAllBytes(Path.Combine(outputDirAbsolute, fileName), bytes);

                atlasFileByPage[pageKvp.Key] = outputDirRelativeToJson.Replace('\\', '/') + "/" + fileName;

                UnityEngine.Object.DestroyImmediate(tex);
            }

            var result = new LitboxDemoSceneExporter.TextureAtlasKeyJson[sources.Count];
            int idx = 0;
            foreach (var kvp in sources)
            {
                var src = kvp.Value;
                var placement = placementsByName[kvp.Key];
                var (pageW, pageH) = pageSizes[(src.bucket, placement.page)];

                var matrix = ComputeUvTransform(src.width, src.height, placement.x, placement.y, placement.rotation, pageW, pageH);

                result[idx++] = new LitboxDemoSceneExporter.TextureAtlasKeyJson
                {
                    textureName = kvp.Key,
                    atlasName = atlasFileByPage[(src.bucket, placement.page)],
                    uvTransform = FormatMatrix(matrix)
                };
            }

            return result;
        }

        private static FormatBucket Classify(Texture tex)
        {
            if (!(tex is Texture2D) && !(tex is RenderTexture))
            {
                throw new InvalidOperationException(
                    $"Texture '{tex.name}' has unsupported type {tex.GetType().Name}; only Texture2D and RenderTexture can be atlas-baked.");
            }

            var textureFormat = GraphicsFormatUtility.GetTextureFormat(tex.graphicsFormat);

            // BC1/DXT1 compression is broadly unsupported on mobile GPUs, so the
            // output of this pipeline must stay in the 3 universally-supported
            // formats below. DXT1/DXT1Crunched (and DXT5/DXT5Crunched) all decode
            // losslessly via GetPixels() the same as any other format ForceReadData
            // handles, and are folded into the RGBA32 sRGB bucket.
            //
            // FormatBucket.Dxt1 and its native block-compressed baking path
            // (ExtrudeBlockSlack, EncodeDxt1Container, the block-unit packing branch
            // in BakeAtlases) are kept in this file, dormant, in case a desktop-only
            // export path is wanted later - Classify() simply never returns it.
            bool isRgba32Like = textureFormat == TextureFormat.RGBA32
                || textureFormat == TextureFormat.RGB24
                || textureFormat == TextureFormat.DXT1
                || textureFormat == TextureFormat.DXT1Crunched
                || textureFormat == TextureFormat.DXT5
                || textureFormat == TextureFormat.DXT5Crunched;

            if (isRgba32Like)
            {
                if (!tex.isDataSRGB)
                {
                    throw new InvalidOperationException(
                        $"Texture '{tex.name}' ({textureFormat}) is not sRGB; only sRGB RGBA32/RGB24/DXT1/DXT5 textures are supported for atlas baking.");
                }
                // RGB24 has no alpha channel; ForceReadData() reads it via GetPixels(),
                // which reports alpha=1 for RGB24 source data, so it folds into the
                // RGBA32 bucket with no separate conversion step needed here.
                return FormatBucket.Rgba32Srgb;
            }

            if (textureFormat == TextureFormat.RFloat) return FormatBucket.RFloat;
            if (textureFormat == TextureFormat.RGBAFloat) return FormatBucket.RgbaFloat;

            throw new InvalidOperationException(
                $"Texture '{tex.name}' has unsupported format {textureFormat}; only sRGB RGBA32/RGB24/DXT1/DXT5, RFloat, and RGBAFloat are supported for atlas baking.");
        }

        private static int CeilDiv(int a, int b) => (a + b - 1) / b;

        private static Texture2D CreatePageTexture(FormatBucket bucket, int w, int h)
        {
            // Both float buckets are physically stored as RGBAFloat (matching
            // TextureExtensions.SaveTextureEXR's proven encode path); they still land
            // in separate atlas pages/files because grouping happens before this point.
            // Dxt1 is assembled as an uncompressed RGBA32 buffer too, then compressed
            // to real DXT1 in-place right before encoding (see EncodeDxt1Container).
            return bucket == FormatBucket.Rgba32Srgb || bucket == FormatBucket.Dxt1
                ? new Texture2D(w, h, TextureFormat.RGBA32, false, false)
                : new Texture2D(w, h, TextureFormat.RGBAFloat, false, true);
        }

        private static string BucketSlug(FormatBucket bucket) => bucket switch
        {
            FormatBucket.Rgba32Srgb => "rgba32_srgb",
            FormatBucket.RFloat => "rfloat",
            FormatBucket.RgbaFloat => "rgba_float",
            FormatBucket.Dxt1 => "dxt1",
            _ => throw new InvalidOperationException($"Unknown format bucket {bucket}")
        };

        private static string BucketExtension(FormatBucket bucket) => bucket switch
        {
            FormatBucket.Rgba32Srgb => "png",
            FormatBucket.Dxt1 => "bc1",
            _ => "exr"
        };

        // Compresses the assembled RGBA32 page texture to real DXT1 (in place) and
        // writes it as a minimal custom container: a 20-byte little-endian header
        // (magic "BC11", version, reserved, width, height, block-data length) followed
        // by the tightly-packed raw BC1 block bytes, for an external WebGPU/TypeScript
        // loader to upload directly as a bc1-rgba-unorm-srgb texture.
        //
        // ASSUMPTION (unverified - see plan section G): GetRawTextureData()'s block
        // row order is assumed to match GetPixels()/SetPixels()'s bottom-left-origin
        // convention used everywhere else in this pipeline. If empirical testing shows
        // the loaded image is vertically flipped, fix it here by reversing the
        // block-row iteration direction below (whole 8-byte blocks relocate between
        // row-slots; no internal byte reordering is needed) - nothing upstream of this
        // function needs to change either way.
        private static byte[] EncodeDxt1Container(Texture2D tex, int w, int h)
        {
            EditorUtility.CompressTexture(tex, TextureFormat.DXT1, TextureCompressionQuality.Best);
            byte[] rawBlocks = tex.GetRawTextureData();

            int blocksWide = w / 4;
            int blocksHigh = h / 4;
            long expectedLength = (long)blocksWide * blocksHigh * 8;
            if (rawBlocks.LongLength != expectedLength)
            {
                throw new InvalidOperationException(
                    $"Unexpected DXT1 raw data length for a {w}x{h} atlas page: got {rawBlocks.LongLength} bytes, expected {expectedLength}.");
            }

            using var ms = new MemoryStream(20 + rawBlocks.Length);
            using (var writer = new BinaryWriter(ms))
            {
                writer.Write((byte)'B'); writer.Write((byte)'C'); writer.Write((byte)'1'); writer.Write((byte)'1');
                writer.Write((ushort)1); // version
                writer.Write((ushort)0); // reserved
                writer.Write((uint)w);
                writer.Write((uint)h);
                writer.Write((uint)rawBlocks.Length);
                writer.Write(rawBlocks);
            }

            return ms.ToArray();
        }

        // Clamp-fills the 0-3px block-rounding slack on the +X/+Y sides of a placed
        // Dxt1 texture's real content (content is always anchored at the same corner
        // as its block-rounded footprint, so slack never occurs on -X/-Y). Must run
        // before the outer padding-ring extrusion, which then treats the whole
        // block-rounded region as "the content".
        private static void ExtrudeBlockSlack(Color[] buf, int pageW, int destX, int destY, int contentW, int contentH, int slackRight, int slackTop)
        {
            for (int p = 0; p < slackRight; p++)
            {
                int x = destX + contentW + p;
                for (int y = destY; y < destY + contentH; y++)
                    buf[x + y * pageW] = buf[(destX + contentW - 1) + y * pageW];
            }

            for (int p = 0; p < slackTop; p++)
            {
                int y = destY + contentH + p;
                for (int x = destX; x < destX + contentW; x++)
                    buf[x + y * pageW] = buf[x + (destY + contentH - 1) * pageW];
            }

            if (slackRight > 0 && slackTop > 0)
            {
                Color corner = buf[(destX + contentW - 1) + (destY + contentH - 1) * pageW];
                for (int py = 0; py < slackTop; py++)
                {
                    int y = destY + contentH + py;
                    for (int px = 0; px < slackRight; px++)
                    {
                        int x = destX + contentW + px;
                        buf[x + y * pageW] = corner;
                    }
                }
            }
        }

        private static Color ToColor(float4 v) => new Color(v.x, v.y, v.z, v.w);

        // Writes src (rawSrcW x rawSrcH, bottom-left origin) into dst at (destX, destY),
        // rotated CCW by `rotation` degrees. Must match ComputeUvTransform's convention exactly.
        private static void BlitRotated(float4[,] src, int srcW, int srcH, Color[] dst, int dstPageW, int destX, int destY, int rotation)
        {
            for (int j = 0; j < srcH; j++)
            {
                for (int i = 0; i < srcW; i++)
                {
                    int ax, ay;
                    switch (rotation)
                    {
                        case 90:
                            ax = destX + (srcH - 1 - j);
                            ay = destY + i;
                            break;
                        case 180:
                            ax = destX + (srcW - 1 - i);
                            ay = destY + (srcH - 1 - j);
                            break;
                        case 270:
                            ax = destX + j;
                            ay = destY + (srcW - 1 - i);
                            break;
                        default:
                            ax = destX + i;
                            ay = destY + j;
                            break;
                    }
                    dst[ax + ay * dstPageW] = ToColor(src[i, j]);
                }
            }
        }

        // Clamp-extrudes the placed content's boundary pixels into the surrounding
        // padding (including corners) so bilinear sampling at a UV edge never bleeds
        // into a packing neighbor. Must run after rotation, in atlas space.
        private static void ExtrudeEdges(Color[] buf, int pageW, int pageH, int destX, int destY, int contentW, int contentH, int padding)
        {
            for (int p = 1; p <= padding; p++)
            {
                int x = destX - p;
                if (x >= 0)
                {
                    for (int y = destY; y < destY + contentH; y++)
                        buf[x + y * pageW] = buf[destX + y * pageW];
                }
            }

            for (int p = 0; p < padding; p++)
            {
                int x = destX + contentW + p;
                if (x >= pageW) break;
                for (int y = destY; y < destY + contentH; y++)
                    buf[x + y * pageW] = buf[(destX + contentW - 1) + y * pageW];
            }

            for (int p = 1; p <= padding; p++)
            {
                int y = destY - p;
                if (y >= 0)
                {
                    for (int x = destX; x < destX + contentW; x++)
                        buf[x + y * pageW] = buf[x + destY * pageW];
                }
            }

            for (int p = 0; p < padding; p++)
            {
                int y = destY + contentH + p;
                if (y >= pageH) break;
                for (int x = destX; x < destX + contentW; x++)
                    buf[x + y * pageW] = buf[x + (destY + contentH - 1) * pageW];
            }

            for (int py = 1; py <= padding; py++)
            {
                for (int px = 1; px <= padding; px++)
                {
                    int x = destX - px, y = destY - py;
                    if (x >= 0 && y >= 0) buf[x + y * pageW] = buf[destX + destY * pageW];

                    x = destX + contentW - 1 + px; y = destY - py;
                    if (x < pageW && y >= 0) buf[x + y * pageW] = buf[(destX + contentW - 1) + destY * pageW];

                    x = destX - px; y = destY + contentH - 1 + py;
                    if (x >= 0 && y < pageH) buf[x + y * pageW] = buf[destX + (destY + contentH - 1) * pageW];

                    x = destX + contentW - 1 + px; y = destY + contentH - 1 + py;
                    if (x < pageW && y < pageH) buf[x + y * pageW] = buf[(destX + contentW - 1) + (destY + contentH - 1) * pageW];
                }
            }
        }

        // Maps a source texture's original UV in [0,1]^2 (Unity convention: origin
        // bottom-left, V up) to the atlas UV, for the given placement/rotation.
        private static Matrix4x4 ComputeUvTransform(int srcW, int srcH, int destX, int destY, int rotation, int atlasW, int atlasH)
        {
            var m = Matrix4x4.identity;

            switch (rotation)
            {
                case 0:
                    m.m00 = (float)srcW / atlasW; m.m01 = 0; m.m02 = (float)destX / atlasW;
                    m.m10 = 0; m.m11 = (float)srcH / atlasH; m.m12 = (float)destY / atlasH;
                    break;
                case 180:
                    m.m00 = -(float)srcW / atlasW; m.m01 = 0; m.m02 = (float)(destX + srcW) / atlasW;
                    m.m10 = 0; m.m11 = -(float)srcH / atlasH; m.m12 = (float)(destY + srcH) / atlasH;
                    break;
                case 90:
                    m.m00 = 0; m.m01 = -(float)srcH / atlasW; m.m02 = (float)(destX + srcH) / atlasW;
                    m.m10 = (float)srcW / atlasH; m.m11 = 0; m.m12 = (float)destY / atlasH;
                    break;
                case 270:
                    m.m00 = 0; m.m01 = (float)srcH / atlasW; m.m02 = (float)destX / atlasW;
                    m.m10 = -(float)srcW / atlasH; m.m11 = 0; m.m12 = (float)(destY + srcW) / atlasH;
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported rotation {rotation}");
            }

            return m;
        }

        private static string FormatMatrix(Matrix4x4 m)
        {
            return $"[[{m.m00}, {m.m01}, {m.m02}], [{m.m10}, {m.m11}, {m.m12}]]";
        }
    }
}
