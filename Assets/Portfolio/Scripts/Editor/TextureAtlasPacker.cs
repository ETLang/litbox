using System;
using System.Collections.Generic;
using System.Linq;

namespace Litbox.Portfolio
{
    public struct PackItem
    {
        public string id;
        public int rawWidth;
        public int rawHeight;
    }

    public struct Placement
    {
        public string id;
        public int page;
        public int x; // content origin (padding excluded)
        public int y; // content origin (padding excluded)
        public int rotation; // 0, 90, 180, 270 (CCW)
        public int pageWidth;
        public int pageHeight;
    }

    // Pure MaxRects (Best Short Side Fit) bin packer. No Unity dependency.
    // Never resizes input rects; only considers 0/90 degree orientations (180/270
    // have the same bounding box as 0/90, so they add no packing benefit here).
    public static class TextureAtlasPacker
    {
        public static List<Placement> Pack(IList<PackItem> items, int padding, int maxPageSize)
        {
            var result = new List<Placement>();
            var remaining = SortItems(items);
            int page = 0;

            while (remaining.Count > 0)
            {
                int pageSize = InitialGuess(remaining, padding, maxPageSize);
                List<WorkPlacement> placed;
                List<PackItem> leftover;

                while (true)
                {
                    (placed, leftover) = PackGreedy(remaining, padding, pageSize, pageSize);
                    if (leftover.Count == 0 || pageSize >= maxPageSize) break;
                    pageSize = Math.Min(pageSize * 2, maxPageSize);
                }

                if (leftover.Count == remaining.Count)
                {
                    var culprit = remaining[0];
                    throw new InvalidOperationException(
                        $"Texture '{culprit.id}' ({culprit.rawWidth}x{culprit.rawHeight}) does not fit within the maximum atlas page size ({maxPageSize}x{maxPageSize}), even alone.");
                }

                foreach (var p in placed)
                {
                    result.Add(new Placement
                    {
                        id = p.id,
                        page = page,
                        x = p.x,
                        y = p.y,
                        rotation = p.rotation,
                        pageWidth = pageSize,
                        pageHeight = pageSize
                    });
                }

                remaining = leftover;
                page++;
            }

            return result;
        }

        private struct WorkPlacement
        {
            public string id;
            public int x;
            public int y;
            public int rotation;
        }

        private struct RectI
        {
            public int X, Y, Width, Height;
            public RectI(int x, int y, int w, int h) { X = x; Y = y; Width = w; Height = h; }
            public int Right => X + Width;
            public int Top => Y + Height;
        }

        private static (List<WorkPlacement> placed, List<PackItem> leftover) PackGreedy(
            List<PackItem> items, int padding, int pageW, int pageH)
        {
            var freeRects = new List<RectI> { new RectI(0, 0, pageW, pageH) };
            var placed = new List<WorkPlacement>();
            var leftover = new List<PackItem>();

            foreach (var item in items)
            {
                int w0 = item.rawWidth + 2 * padding;
                int h0 = item.rawHeight + 2 * padding;
                int w90 = item.rawHeight + 2 * padding;
                int h90 = item.rawWidth + 2 * padding;
                bool canRotate = item.rawWidth != item.rawHeight;

                int bestFreeIndex = -1;
                int bestRotation = 0;
                int bestShortSideFit = int.MaxValue;
                int bestLongSideFit = int.MaxValue;

                for (int i = 0; i < freeRects.Count; i++)
                {
                    var free = freeRects[i];

                    if (w0 <= free.Width && h0 <= free.Height)
                    {
                        int lw = free.Width - w0;
                        int lh = free.Height - h0;
                        int shortSide = Math.Min(lw, lh);
                        int longSide = Math.Max(lw, lh);
                        if (shortSide < bestShortSideFit || (shortSide == bestShortSideFit && longSide < bestLongSideFit))
                        {
                            bestShortSideFit = shortSide;
                            bestLongSideFit = longSide;
                            bestFreeIndex = i;
                            bestRotation = 0;
                        }
                    }

                    if (canRotate && w90 <= free.Width && h90 <= free.Height)
                    {
                        int lw = free.Width - w90;
                        int lh = free.Height - h90;
                        int shortSide = Math.Min(lw, lh);
                        int longSide = Math.Max(lw, lh);
                        if (shortSide < bestShortSideFit || (shortSide == bestShortSideFit && longSide < bestLongSideFit))
                        {
                            bestShortSideFit = shortSide;
                            bestLongSideFit = longSide;
                            bestFreeIndex = i;
                            bestRotation = 90;
                        }
                    }
                }

                if (bestFreeIndex < 0)
                {
                    leftover.Add(item);
                    continue;
                }

                var chosen = freeRects[bestFreeIndex];
                int placedW = bestRotation == 0 ? w0 : w90;
                int placedH = bestRotation == 0 ? h0 : h90;
                var placedRect = new RectI(chosen.X, chosen.Y, placedW, placedH);

                placed.Add(new WorkPlacement
                {
                    id = item.id,
                    x = placedRect.X + padding,
                    y = placedRect.Y + padding,
                    rotation = bestRotation
                });

                SplitFreeRects(freeRects, placedRect);
                PruneFreeRects(freeRects);
            }

            return (placed, leftover);
        }

        private static bool Intersects(RectI a, RectI b)
        {
            return a.X < b.Right && a.Right > b.X && a.Y < b.Top && a.Top > b.Y;
        }

        private static void SplitFreeRects(List<RectI> freeRects, RectI placed)
        {
            for (int i = freeRects.Count - 1; i >= 0; i--)
            {
                var free = freeRects[i];
                if (!Intersects(free, placed)) continue;

                freeRects.RemoveAt(i);

                if (placed.X > free.X)
                {
                    freeRects.Add(new RectI(free.X, free.Y, placed.X - free.X, free.Height));
                }
                if (placed.Right < free.Right)
                {
                    freeRects.Add(new RectI(placed.Right, free.Y, free.Right - placed.Right, free.Height));
                }
                if (placed.Y > free.Y)
                {
                    freeRects.Add(new RectI(free.X, free.Y, free.Width, placed.Y - free.Y));
                }
                if (placed.Top < free.Top)
                {
                    freeRects.Add(new RectI(free.X, placed.Top, free.Width, free.Top - placed.Top));
                }
            }
        }

        private static void PruneFreeRects(List<RectI> freeRects)
        {
            for (int i = freeRects.Count - 1; i >= 0; i--)
            {
                for (int j = 0; j < freeRects.Count; j++)
                {
                    if (i == j) continue;
                    if (Contains(freeRects[j], freeRects[i]))
                    {
                        freeRects.RemoveAt(i);
                        break;
                    }
                }
            }
        }

        private static bool Contains(RectI a, RectI b)
        {
            return b.X >= a.X && b.Y >= a.Y && b.Right <= a.Right && b.Top <= a.Top;
        }

        private static List<PackItem> SortItems(IList<PackItem> items)
        {
            return items
                .OrderByDescending(i => Math.Max(i.rawWidth, i.rawHeight))
                .ThenByDescending(i => Math.Min(i.rawWidth, i.rawHeight))
                .ThenBy(i => i.id, StringComparer.Ordinal)
                .ToList();
        }

        private static int InitialGuess(List<PackItem> items, int padding, int maxPageSize)
        {
            long totalArea = 0;
            foreach (var item in items)
            {
                long w = item.rawWidth + 2L * padding;
                long h = item.rawHeight + 2L * padding;
                totalArea += w * h;
            }

            int guess = NextPowerOfTwo((int)Math.Ceiling(Math.Sqrt(totalArea * 1.2)));
            return Math.Clamp(guess, 64, maxPageSize);
        }

        private static int NextPowerOfTwo(int v)
        {
            if (v < 1) return 1;
            int p = 1;
            while (p < v) p <<= 1;
            return p;
        }
    }
}
