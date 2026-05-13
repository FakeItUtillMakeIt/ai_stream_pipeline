#include <vector>
#include <utility>
#include <cmath>
#include <set>

typedef struct PixelPoint
{
    float x;
    float y;
    PixelPoint() : x(0.0f), y(0.0f){}
    PixelPoint(float x, float y) : x(x), y(y) {}
} PixelPoint;

class ZoneValidator
{
public:
    static bool zoneIsValid(
        const std::vector<PixelPoint> &zone_points)
    {
        // =====================================================
        // 1. 至少3个点
        // =====================================================

        if (zone_points.size() < 3)
        {
            return false;
        }

        // =====================================================
        // 2. 检查 NaN / Inf
        // =====================================================

        for (const auto &p : zone_points)
        {
            if (!std::isfinite(p.x) ||
                !std::isfinite(p.y))
            {
                return false;
            }
        }

        // =====================================================
        // 3. 检查重复点
        // =====================================================

        {
            std::set<std::pair<int, int>> unique_points;

            constexpr float scale = 10000.0f;

            for (const auto &p : zone_points)
            {
                int x = static_cast<int>(p.x * scale);
                int y = static_cast<int>(p.y * scale);

                auto key = std::make_pair(x, y);

                if (unique_points.count(key))
                {
                    return false;
                }

                unique_points.insert(key);
            }
        }

        // =====================================================
        // 4. 面积检查（Shoelace Formula）
        // =====================================================

        float area = polygonArea(zone_points);

        constexpr float MIN_AREA = 1e-3f;

        if (std::fabs(area) < MIN_AREA)
        {
            return false;
        }

        // =====================================================
        // 5. 检查所有点是否共线
        // =====================================================

        if (isCollinear(zone_points))
        {
            return false;
        }

        // =====================================================
        // 6. 检查自相交
        // =====================================================

        if (isSelfIntersecting(zone_points))
        {
            return false;
        }

        return true;
    }

    static std::vector<std::pair<float, float>> getGlobalZone(
        int image_width,
        int image_height)
    {
        std::vector<std::pair<float, float>> zone;

        zone.emplace_back(0.0f, 0.0f);

        zone.emplace_back(
            static_cast<float>(image_width),
            0.0f);

        zone.emplace_back(
            static_cast<float>(image_width),
            static_cast<float>(image_height));

        zone.emplace_back(
            0.0f,
            static_cast<float>(image_height));

        return zone;
    }

    static bool pointInPolygon(
        const PixelPoint& point,
        const std::vector<PixelPoint>& polygon,
        bool include_boundary = true)
    {
        const size_t n = polygon.size();

        if (n < 3)
        {
            return false;
        }

        const float px = point.x;
        const float py = point.y;

        bool inside = false;

        for (size_t i = 0, j = n - 1; i < n; j = i++)
        {
            const float xi = polygon[i].x;
            const float yi = polygon[i].y;

            const float xj = polygon[j].x;
            const float yj = polygon[j].y;

            // =================================================
            // 1. 边界判断
            // =================================================

            if (include_boundary)
            {
                if (pointOnSegment(
                        point,
                        polygon[j],
                        polygon[i]))
                {
                    return true;
                }
            }

            // =================================================
            // 2. Ray Casting
            // =================================================

            const bool intersect =
                ((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12f) + xi);

            if (intersect)
            {
                inside = !inside;
            }
        }

        return inside;
    }
private:
    // =========================================================
    // 多边形面积
    // =========================================================

    static float polygonArea(
        const std::vector<PixelPoint> &pts)
    {
        float area = 0.0f;

        int n = static_cast<int>(pts.size());

        for (int i = 0; i < n; ++i)
        {
            int j = (i + 1) % n;

            area += pts[i].x * pts[j].y;
            area -= pts[j].x * pts[i].y;
        }

        return area * 0.5f;
    }

    // =========================================================
    // 判断是否共线
    // =========================================================

    static bool isCollinear(
        const std::vector<PixelPoint> &pts)
    {
        if (pts.size() < 3)
        {
            return true;
        }

        const auto &p0 = pts[0];
        const auto &p1 = pts[1];

        constexpr float EPS = 1e-6f;

        for (size_t i = 2; i < pts.size(); ++i)
        {
            const auto &p = pts[i];

            float cross =
                (p1.x - p0.x) * (p.y - p0.y) -
                (p1.y - p0.y) * (p.x - p0.x);

            if (std::fabs(cross) > EPS)
            {
                return false;
            }
        }

        return true;
    }

    // =========================================================
    // 叉积方向
    // =========================================================

    static int orientation(
        const PixelPoint &p,
        const PixelPoint &q,
        const PixelPoint &r)
    {
        float val =
            (q.y - p.y) * (r.x - q.x) -
            (q.x - p.x) * (r.y - q.y);

        constexpr float EPS = 1e-6f;

        if (std::fabs(val) < EPS)
        {
            return 0;
        }

        return (val > 0.0f) ? 1 : 2;
    }

    // =========================================================
    // 判断点在线段上
    // =========================================================

    static bool pointOnSegment(
        const PixelPoint &p,
        const PixelPoint &q,
        const PixelPoint &r)
    {
        return q.x <= std::max(p.x, r.x) &&
               q.x >= std::min(p.x, r.x) &&
               q.y <= std::max(p.y, r.y) &&
               q.y >= std::min(p.y, r.y);
    }

    // =========================================================
    // 判断两线段是否相交
    // =========================================================

    static bool segmentsIntersect(
        const PixelPoint &p1,
        const PixelPoint &q1,
        const PixelPoint &p2,
        const PixelPoint &q2)
    {
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);

        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);

        if (o1 != o2 && o3 != o4)
        {
            return true;
        }

        // 特殊情况

        if (o1 == 0 && pointOnSegment(p1, p2, q1))
        {
            return true;
        }

        if (o2 == 0 && pointOnSegment(p1, q2, q1))
        {
            return true;
        }

        if (o3 == 0 && pointOnSegment(p2, p1, q2))
        {
            return true;
        }

        if (o4 == 0 && pointOnSegment(p2, q1, q2))
        {
            return true;
        }

        return false;
    }

    // =========================================================
    // 检查多边形是否自相交
    // =========================================================

    static bool isSelfIntersecting(
        const std::vector<PixelPoint> &pts)
    {
        int n = static_cast<int>(pts.size());

        for (int i = 0; i < n; ++i)
        {
            auto a1 = pts[i];
            auto a2 = pts[(i + 1) % n];

            for (int j = i + 1; j < n; ++j)
            {
                // 跳过相邻边

                if (std::abs(i - j) <= 1)
                {
                    continue;
                }

                // 跳过首尾边

                if (i == 0 && j == n - 1)
                {
                    continue;
                }

                auto b1 = pts[j];
                auto b2 = pts[(j + 1) % n];

                if (segmentsIntersect(a1, a2, b1, b2))
                {
                    return true;
                }
            }
        }

        return false;
    }
};