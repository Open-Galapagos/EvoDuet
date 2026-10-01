// EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

using u64 = uint64_t;

struct Point {
    double x, y;
};

struct Segment {
    double ax, ay, dx, dy, inv_len2;
};

struct RNG {
    u64 x;
    explicit RNG(u64 seed) : x(seed) {}
    u64 next() {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return x * 2685821657736338717ULL;
    }
    long double uniform() { return ldexpl((long double)next(), -64); }
};

static inline float point_segment_distance(const Point &p, const Segment &s) {
    double px = p.x - s.ax, py = p.y - s.ay;
    double t = (px * s.dx + py * s.dy) * s.inv_len2;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    double ex = px - t * s.dx, ey = py - t * s.dy;
    return (float)sqrt(ex * ex + ey * ey);
}

template<class Heap>
static inline void keep_smallest(Heap &h, pair<double,int> item, int cap) {
    if ((int)h.size() < cap) h.push(item);
    else if (item < h.top()) {
        h.pop();
        h.push(item);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, K, S;
    u64 seed;
    if (!(cin >> n >> m >> K >> S >> seed)) return 0;

    vector<Point> vertex(n);
    double minx = numeric_limits<double>::infinity();
    double miny = minx, maxx = -minx, maxy = -minx;
    for (Point &p : vertex) {
        long long x, y;
        cin >> x >> y;
        p = {(double)x, (double)y};
        minx = min(minx, p.x); maxx = max(maxx, p.x);
        miny = min(miny, p.y); maxy = max(maxy, p.y);
    }

    vector<pair<int,int>> endpoint(m);
    vector<Segment> seg(m);
    for (int i = 0; i < m; ++i) {
        int a, b;
        cin >> a >> b;
        endpoint[i] = {a, b};
        double dx = vertex[b].x - vertex[a].x;
        double dy = vertex[b].y - vertex[a].y;
        double l2 = dx * dx + dy * dy;
        seg[i] = {vertex[a].x, vertex[a].y, dx, dy, l2 > 0 ? 1.0 / l2 : 0.0};
    }

    // Reproduce the checker's points. Long double is used for triangle choice and
    // barycentric generation so even very unbalanced polygons pick identical triangles.
    vector<long double> pref(max(0, n - 2));
    long double total_area = 0;
    for (int i = 1; i + 1 < n; ++i) {
        long double abx = (long double)vertex[i].x - vertex[0].x;
        long double aby = (long double)vertex[i].y - vertex[0].y;
        long double acx = (long double)vertex[i+1].x - vertex[0].x;
        long double acy = (long double)vertex[i+1].y - vertex[0].y;
        total_area += (abx * acy - aby * acx) * 0.5L;
        pref[i-1] = total_area;
    }
    vector<Point> sample(S);
    RNG rng(seed);
    for (int t = 0; t < S; ++t) {
        long double r = rng.uniform() * total_area;
        int z = int(upper_bound(pref.begin(), pref.end(), r) - pref.begin());
        int i = z + 1;
        long double root = sqrtl(rng.uniform());
        long double vv = rng.uniform();
        long double alpha = 1 - root;
        long double beta = root * (1 - vv);
        long double gamma = root * vv;
        sample[t].x = (double)(alpha * vertex[0].x + beta * vertex[i].x + gamma * vertex[i+1].x);
        sample[t].y = (double)(alpha * vertex[0].y + beta * vertex[i].y + gamma * vertex[i+1].y);
    }

    if (K == m) {
        for (int i = 0; i < K; ++i) cout << i << (i + 1 == K ? '\n' : ' ');
        return 0;
    }

    auto start_time = chrono::steady_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
    };

    // Evaluate every rope on a small, spread-out coreset. In addition to globally
    // good ropes, preserve ropes good in each spatial cell and ropes passing very
    // close to individual points; those are often needed late in a K-rope solution.
    int screen_q = min(S, max(96, min(768, 22000000 / max(1, m))));
    vector<int> screen_idx(screen_q), cell(screen_q);
    for (int q = 0; q < screen_q; ++q)
        screen_idx[q] = (int)((long long)q * S / screen_q);
    double rangex = max(1e-30, maxx - minx), rangey = max(1e-30, maxy - miny);
    int cell_count[16] = {};
    for (int q = 0; q < screen_q; ++q) {
        const Point &p = sample[screen_idx[q]];
        int gx = min(3, max(0, int((p.x - minx) / rangex * 4.0)));
        int gy = min(3, max(0, int((p.y - miny) / rangey * 4.0)));
        cell[q] = gy * 4 + gx;
        ++cell_count[cell[q]];
    }

    vector<pair<double,int>> mean_score;
    mean_score.reserve(m);
    vector<priority_queue<pair<double,int>>> cell_best(16);
    static constexpr int NEAR_KEEP = 12;
    int near_keep = (K >= 40 ? NEAR_KEEP : 2);
    vector<array<pair<float,int>,NEAR_KEEP>> nearest(screen_q);
    for (auto &a : nearest) {
        for (auto &v : a) v = {numeric_limits<float>::infinity(), -1};
    }
    for (int id = 0; id < m; ++id) {
        double sum = 0;
        double csum[16] = {};
        for (int q = 0; q < screen_q; ++q) {
            float d = point_segment_distance(sample[screen_idx[q]], seg[id]);
            sum += d;
            csum[cell[q]] += d;
            auto &a = nearest[q];
            pair<float,int> val = {d, id};
            if (val < a[near_keep-1]) {
                int at = near_keep - 1;
                while (at > 0 && val < a[at-1]) { a[at] = a[at-1]; --at; }
                a[at] = val;
            }
        }
        mean_score.push_back({sum / screen_q, id});
        for (int c = 0; c < 16; ++c) if (cell_count[c])
            keep_smallest(cell_best[c], {csum[c] / cell_count[c], id}, 36);
    }

    int mean_keep = min(m, K == 1 ? 16000 : 8000);
    if (mean_keep < m)
        nth_element(mean_score.begin(), mean_score.begin() + mean_keep, mean_score.end());
    sort(mean_score.begin(), mean_score.begin() + mean_keep);

    vector<int> pool;
    pool.reserve(3200);
    vector<unsigned char> in_pool(m, 0);
    auto add_pool = [&](int id) {
        if (id >= 0 && !in_pool[id]) { in_pool[id] = 1; pool.push_back(id); }
    };
    for (int i = 0; i < mean_keep; ++i) add_pool(mean_score[i].second);
    for (auto &a : nearest) for (int z = 0; z < near_keep; ++z) add_pool(a[z].second);
    for (auto &h : cell_best) while (!h.empty()) { add_pool(h.top().second); h.pop(); }
    for (int i = 0; i < K; ++i) add_pool(i); // Never discard the reference set.

    // A few long ropes and an order-independent spread of the input protect against
    // a noisy screening coreset and unusual candidate generators.
    vector<pair<double,int>> lengths;
    lengths.reserve(m);
    for (int id = 0; id < m; ++id) {
        double l2 = seg[id].dx * seg[id].dx + seg[id].dy * seg[id].dy;
        lengths.push_back({-l2, id});
    }
    int long_keep = min(m, 128);
    if (long_keep < m) nth_element(lengths.begin(), lengths.begin() + long_keep, lengths.end());
    for (int i = 0; i < long_keep; ++i) add_pool(lengths[i].second);
    for (int j = 0; j < min(m, 160); ++j)
        add_pool((int)((long long)j * m / min(m, 160)));

    const int P = (int)pool.size();
    int fit_q = min(S, 2600);
    if (P <= 4000)
        fit_q = min(S, max(fit_q, 16000000 / max(1, P)));
    // Keep the distance matrix comfortably below the memory cap.
    fit_q = min(fit_q, max(400, 42000000 / max(1, P)));
    vector<int> fit_idx(fit_q);
    for (int q = 0; q < fit_q; ++q)
        fit_idx[q] = (int)((long long)q * S / fit_q);
    vector<float> distance((size_t)P * fit_q);
    for (int j = 0; j < P; ++j) {
        float *row = distance.data() + (size_t)j * fit_q;
        for (int q = 0; q < fit_q; ++q)
            row[q] = point_segment_distance(sample[fit_idx[q]], seg[pool[j]]);
    }

    vector<int> selected;
    selected.reserve(K);
    vector<unsigned char> used(P, 0);
    vector<float> best(fit_q, numeric_limits<float>::infinity());

    // First greedy choice.
    int first = 0;
    double first_sum = numeric_limits<double>::infinity();
    for (int j = 0; j < P; ++j) {
        const float *row = distance.data() + (size_t)j * fit_q;
        double sum = 0;
        for (int q = 0; q < fit_q; ++q) sum += row[q];
        if (sum < first_sum) { first_sum = sum; first = j; }
    }
    selected.push_back(first); used[first] = 1;
    copy_n(distance.data() + (size_t)first * fit_q, fit_q, best.begin());

    struct GainNode {
        double gain;
        int id, stamp;
        bool operator<(const GainNode &o) const { return gain < o.gain; }
    };
    priority_queue<GainNode> gains;
    for (int j = 0; j < P; ++j) if (!used[j]) {
        const float *row = distance.data() + (size_t)j * fit_q;
        double gain = 0;
        for (int q = 0; q < fit_q; ++q) if (row[q] < best[q]) gain += best[q] - row[q];
        gains.push({gain, j, 1});
    }

    // Lazy submodular greedy: old marginal gains are valid upper bounds.
    while ((int)selected.size() < K && !gains.empty()) {
        int stamp = (int)selected.size();
        GainNode node = gains.top(); gains.pop();
        if (used[node.id]) continue;
        if (node.stamp != stamp) {
            const float *row = distance.data() + (size_t)node.id * fit_q;
            double gain = 0;
            for (int q = 0; q < fit_q; ++q) if (row[q] < best[q]) gain += best[q] - row[q];
            gains.push({gain, node.id, stamp});
            continue;
        }
        selected.push_back(node.id); used[node.id] = 1;
        const float *row = distance.data() + (size_t)node.id * fit_q;
        for (int q = 0; q < fit_q; ++q) best[q] = min(best[q], row[q]);
    }
    for (int j = 0; (int)selected.size() < K && j < P; ++j)
        if (!used[j]) { used[j] = 1; selected.push_back(j); }

    // Best-improvement 1-swap local search. For one proposed incoming rope, all K
    // possible removals are evaluated in one pass by charging only points owned by
    // the removed rope.
    vector<float> second(fit_q);
    vector<int> owner(fit_q);
    auto rebuild = [&]() -> double {
        double cost = 0;
        for (int q = 0; q < fit_q; ++q) {
            float b1 = numeric_limits<float>::infinity();
            float b2 = numeric_limits<float>::infinity();
            int own = 0;
            for (int r = 0; r < K; ++r) {
                float d = distance[(size_t)selected[r] * fit_q + q];
                if (d < b1) { b2 = b1; b1 = d; own = r; }
                else if (d < b2) b2 = d;
            }
            best[q] = b1; second[q] = b2; owner[q] = own; cost += b1;
        }
        return cost;
    };
    double current_cost = rebuild();
    auto improve_by_swaps = [&](int limit) {
        int swaps = 0;
        while (swaps < limit && elapsed() < 1.68) {
            double new_cost = current_cost;
            int add = -1, remove_pos = -1;
            vector<double> adjustment(K);
            for (int j = 0; j < P; ++j) if (!used[j]) {
                fill(adjustment.begin(), adjustment.end(), 0.0);
                const float *row = distance.data() + (size_t)j * fit_q;
                double add_cost = 0;
                for (int q = 0; q < fit_q; ++q) {
                    float incoming = row[q];
                    float with_add = min(best[q], incoming);
                    add_cost += with_add;
                    adjustment[owner[q]] += min(second[q], incoming) - with_add;
                }
                for (int r = 0; r < K; ++r) {
                    double cost = add_cost + adjustment[r];
                    if (cost < new_cost) { new_cost = cost; add = j; remove_pos = r; }
                }
            }
            if (add < 0 || new_cost >= current_cost - 1e-7 * max(1.0, current_cost)) break;
            int gone = selected[remove_pos];
            used[gone] = 0; used[add] = 1;
            selected[remove_pos] = add;
            current_cost = rebuild();
            ++swaps;
        }
    };
    improve_by_swaps(18);

    // Pair-removal kicks cross barriers that a pure 1-swap descent cannot. Refilling
    // temporarily bans the removed pair, then ordinary descent is allowed to bring
    // either one back if it belongs in a genuinely better basin.
    if (K >= 2) {
        vector<int> incumbent = selected;
        double incumbent_cost = current_cost;
        int trial = 0;
        while (elapsed() < 1.58 && trial < 12) {
            selected = incumbent;
            fill(used.begin(), used.end(), 0);
            for (int j : selected) used[j] = 1;
            uint64_t h = 0x9e3779b97f4a7c15ULL * (uint64_t)(trial + 1);
            h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 31;
            int p1 = (int)(h % K);
            int p2 = (int)((h >> 32) % (K - 1));
            if (p2 >= p1) ++p2;
            int banned1 = selected[p1], banned2 = selected[p2];
            used[banned1] = used[banned2] = 0;
            selected[p1] = selected[p2] = -1;

            int positions[2] = {p1, p2};
            bool failed = false;
            for (int z = 0; z < 2; ++z) {
                vector<float> partial(fit_q, numeric_limits<float>::infinity());
                for (int r = 0; r < K; ++r) if (selected[r] >= 0) {
                    const float *row = distance.data() + (size_t)selected[r] * fit_q;
                    for (int q = 0; q < fit_q; ++q) partial[q] = min(partial[q], row[q]);
                }
                int choice = -1;
                double choice_cost = numeric_limits<double>::infinity();
                for (int j = 0; j < P; ++j) if (!used[j] && j != banned1 && j != banned2) {
                    const float *row = distance.data() + (size_t)j * fit_q;
                    double cost = 0;
                    for (int q = 0; q < fit_q; ++q) cost += min(partial[q], row[q]);
                    if (cost < choice_cost) { choice_cost = cost; choice = j; }
                }
                if (choice < 0) { failed = true; break; }
                selected[positions[z]] = choice;
                used[choice] = 1;
            }
            if (failed) break;
            current_cost = rebuild();
            improve_by_swaps(10);
            if (current_cost + 1e-7 * max(1.0, incumbent_cost) < incumbent_cost) {
                incumbent = selected;
                incumbent_cost = current_cost;
            }
            ++trial;
        }
        selected = incumbent;
        current_cost = incumbent_cost;
    }

    // With one rope there is no combinatorial search. Spend the otherwise unused
    // budget checking the strongest shortlist on every checker sample exactly.
    if (K == 1) {
        priority_queue<pair<double,int>> shortlist_heap;
        int shortlist_size = min(P, 10000);
        for (int j = 0; j < P; ++j) {
            const float *row = distance.data() + (size_t)j * fit_q;
            double sum = 0;
            for (int q = 0; q < fit_q; ++q) sum += row[q];
            keep_smallest(shortlist_heap, {sum, j}, shortlist_size);
        }
        int exact_choice = selected[0];
        double exact_cost = 0;
        for (const Point &p : sample)
            exact_cost += point_segment_distance(p, seg[pool[exact_choice]]);
        while (!shortlist_heap.empty()) {
            int j = shortlist_heap.top().second;
            shortlist_heap.pop();
            double cost = 0;
            for (const Point &p : sample) cost += point_segment_distance(p, seg[pool[j]]);
            if (cost < exact_cost) { exact_cost = cost; exact_choice = j; }
        }
        selected[0] = exact_choice;
    }

    for (int i = 0; i < K; ++i)
        cout << pool[selected[i]] << (i + 1 == K ? '\n' : ' ');
    return 0;
}
// EVOLVE-BLOCK-END
