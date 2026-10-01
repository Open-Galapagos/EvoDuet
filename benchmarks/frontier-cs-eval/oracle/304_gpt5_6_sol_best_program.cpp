// EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

using int64 = long long;
using i128 = __int128_t;

struct TourResult {
    int64 value;
    vector<int> walk;
};

static TourResult buildTour(const vector<int>& parent,
                            const vector<int>& bfs_order,
                            const vector<vector<int>>& graph,
                            const vector<int64>& rate, int n, int k,
                            int cheap_neighbor, int triangle_neighbor = -1) {
    vector<vector<int>> children(n + 1);
    for (int v = 2; v <= n; ++v) children[parent[v]].push_back(v);

    vector<int64> weight(n + 1), closed_value(n + 1);
    vector<int> subtree_size(n + 1), spine_child(n + 1, -1);
    vector<array<int64, 2>> open_dp(n + 1);
    vector<array<int, 2>> open_depth(n + 1);
    vector<array<int, 2>> choice_child(n + 1), choice_parity(n + 1);
    int endpoint = 1, endpoint_depth = 0, padding = 0;

    auto configure = [&](bool previsited) -> bool {
        weight = rate;
        weight[1] = 0;
        if (previsited && cheap_neighbor != -1) weight[cheap_neighbor] = 0;
        if (previsited && triangle_neighbor != -1) weight[triangle_neighbor] = 0;
        fill(subtree_size.begin(), subtree_size.end(), 1);
        for (int i = n - 1; i > 0; --i) {
            int v = bfs_order[i];
            weight[parent[v]] += weight[v];
            subtree_size[parent[v]] += subtree_size[v];
        }
        for (int v = 1; v <= n; ++v) {
            sort(children[v].begin(), children[v].end(), [&](int a, int b) {
                i128 lhs = (i128)weight[a] * subtree_size[b];
                i128 rhs = (i128)weight[b] * subtree_size[a];
                if (lhs != rhs) return lhs < rhs;
                return a < b;
            });
        }

        const int64 lambda = weight[1] +
                             (previsited && triangle_neighbor != -1
                                  ? rate[triangle_neighbor] : 0);
        const int64 NEG = numeric_limits<int64>::min() / 4;
        for (int oi = n - 1; oi >= 1; --oi) {
            int v = bfs_order[oi];
            int d = (int)children[v].size();
            vector<int64> suffix(d + 1, 0);
            for (int i = d - 1; i >= 0; --i)
                suffix[i] = suffix[i + 1] + weight[children[v][i]];

            int64 base = rate[v];
            if (previsited && (v == cheap_neighbor || v == triangle_neighbor)) base = 0;
            int elapsed = 1;
            for (int ch : children[v]) {
                base += closed_value[ch] + weight[ch] * elapsed;
                elapsed += 2 * subtree_size[ch];
            }
            closed_value[v] = base;
            open_dp[v] = {NEG, base + lambda};
            open_depth[v] = {-1, 1};
            choice_child[v] = {-2, -1};
            choice_parity[v] = {-1, -1};

            int prefix = 0;
            int total_children_length = elapsed - 1;
            for (int i = 0; i < d; ++i) {
                int ch = children[v][i];
                int len = 2 * subtree_size[ch];
                int base_start = 1 + prefix;
                int last_start = 1 + total_children_length - len;
                int64 without = base - closed_value[ch] - weight[ch] * base_start
                                - suffix[i + 1] * len;
                for (int cp = 0; cp < 2; ++cp) {
                    int parity = 1 - cp;
                    int64 cand = without + open_dp[ch][cp]
                                 + weight[ch] * last_start + lambda;
                    if (cand > open_dp[v][parity]) {
                        open_dp[v][parity] = cand;
                        open_depth[v][parity] = 1 + open_depth[ch][cp];
                        choice_child[v][parity] = ch;
                        choice_parity[v][parity] = cp;
                    }
                }
                prefix += len;
            }
        }

        // Schedule all root subtrees closed, then try each one as the open
        // final subtree.  open_dp already includes lambda * endpoint depth.
        int root_degree = (int)children[1].size();
        vector<int64> suffix(root_degree + 1, 0);
        for (int i = root_degree - 1; i >= 0; --i)
            suffix[i] = suffix[i + 1] + weight[children[1][i]];
        int64 root_base = 0;
        int elapsed = 0;
        for (int ch : children[1]) {
            root_base += closed_value[ch] + weight[ch] * elapsed;
            elapsed += 2 * subtree_size[ch];
        }

        int64 best_augmented = NEG;
        int root_choice = -1, root_state = 0;
        auto consider = [&](int64 augmented, int depth, int ch, int cp) {
            int raw_padding = k - (2 * (n - 1) - depth);
            if (triangle_neighbor != -1) {
                if (!previsited || raw_padding < 3 || !(raw_padding & 1)) return;
            } else {
                int candidate_padding = raw_padding & ~1;
                if ((candidate_padding >= 2) != previsited) return;
                augmented -= lambda * (raw_padding & 1);
            }
            if (augmented > best_augmented) {
                best_augmented = augmented;
                root_choice = ch;
                root_state = cp;
            }
        };

        // Root itself is a possible endpoint (no saved return edges).
        if ((!previsited && ((k - 2 * (n - 1)) & ~1) < 2) ||
            (previsited && ((k - 2 * (n - 1)) & ~1) >= 2))
            consider(root_base, 0, -1, 0);

        int prefix = 0;
        for (int i = 0; i < root_degree; ++i) {
            int ch = children[1][i];
            int len = 2 * subtree_size[ch];
            int last_start = elapsed - len;
            int64 without = root_base - closed_value[ch] - weight[ch] * prefix
                            - suffix[i + 1] * len;
            for (int cp = 0; cp < 2; ++cp) {
                if (open_dp[ch][cp] == NEG) continue;
                int64 cand = without + open_dp[ch][cp] + weight[ch] * last_start;
                consider(cand, open_depth[ch][cp], ch, cp);
            }
            prefix += len;
        }
        if (root_choice == -1 && best_augmented == NEG) return false;

        // Trace back once to obtain the real depth, reject the configuration
        // if its assumption about an initial padding trip was wrong.
        fill(spine_child.begin(), spine_child.end(), -1);
        endpoint = 1;
        endpoint_depth = 0;
        if (root_choice != -1) {
            spine_child[1] = root_choice;
            int v = root_choice, state = root_state;
            endpoint_depth = 1;
            while (true) {
                int ch = choice_child[v][state];
                if (ch < 0) {
                    endpoint = v;
                    break;
                }
                spine_child[v] = ch;
                state = choice_parity[v][state];
                v = ch;
                ++endpoint_depth;
            }
        }
        int raw_padding = k - (2 * (n - 1) - endpoint_depth);
        padding = triangle_neighbor == -1 ? (raw_padding & ~1) : raw_padding;

        // Put each selected open child last, retaining the optimal relative
        // order of all closed siblings.
        int v = 1;
        while (spine_child[v] != -1) {
            int ch = spine_child[v];
            auto it = find(children[v].begin(), children[v].end(), ch);
            children[v].erase(it);
            children[v].push_back(ch);
            v = ch;
        }
        return true;
    };

    if (!configure(true) && triangle_neighbor == -1) {
        bool ok = configure(false);
        (void)ok;
    }
    if (triangle_neighbor != -1 && padding < 3) return {-1, {}};

    vector<int> walk;
    walk.reserve(k);
    int even_padding = triangle_neighbor == -1 ? padding : padding - 3;
    for (int used = 0; used < even_padding; used += 2) {
        walk.push_back(cheap_neighbor);
        walk.push_back(1);
    }
    if (triangle_neighbor != -1) {
        walk.push_back(cheap_neighbor);
        walk.push_back(triangle_neighbor);
        walk.push_back(1);
    }

    struct Frame { int v, next_child; };
    vector<Frame> stack;
    stack.reserve(n);
    stack.push_back({1, 0});
    while (!stack.empty()) {
        Frame &f = stack.back();
        if (f.next_child < (int)children[f.v].size()) {
            int to = children[f.v][f.next_child++];
            walk.push_back(to);
            stack.push_back({to, 0});
        } else {
            int v = f.v;
            stack.pop_back();
            if (!stack.empty() && spine_child[parent[v]] != v)
                walk.push_back(parent[v]);
        }
    }
    if ((int)walk.size() < k) {
        int next = endpoint == 1 ? children[1][0] : parent[endpoint];
        walk.push_back(next);
    }

    auto evaluateWalk = [&](const vector<int>& candidate) {
        vector<char> seen(n + 1, false);
        seen[1] = true;
        int64 value = 0;
        for (int t = 1; t <= k; ++t) {
            int v = candidate[t - 1];
            if (!seen[v]) {
                seen[v] = true;
                value += rate[v] * t;
            }
        }
        return value;
    };
    int64 value = evaluateWalk(walk);
    constexpr bool enable_shortcuts = true;
    if (!enable_shortcuts) return {value, move(walk)};

    // Preserve the same preorder, but reach each next new vertex through the
    // closest previously visited neighbor.  The tree path to that neighbor
    // contains only already visited ancestors, so this never changes the
    // first-visit order.
    vector<int> preorder;
    preorder.reserve(n);
    vector<int> todo = {1};
    while (!todo.empty()) {
        int v = todo.back();
        todo.pop_back();
        preorder.push_back(v);
        for (auto it = children[v].rbegin(); it != children[v].rend(); ++it)
            todo.push_back(*it);
    }
    vector<int> rank(n + 1), depth(n + 1, 0);
    for (int i = 0; i < n; ++i) rank[preorder[i]] = i;
    for (int i = 1; i < n; ++i) {
        int v = bfs_order[i];
        depth[v] = depth[parent[v]] + 1;
    }
    int lg = 1;
    while ((1 << lg) <= n) ++lg;
    vector<vector<int>> up(lg, vector<int>(n + 1, 1));
    for (int v = 2; v <= n; ++v) up[0][v] = parent[v];
    for (int j = 1; j < lg; ++j)
        for (int v = 1; v <= n; ++v) up[j][v] = up[j - 1][up[j - 1][v]];
    auto lca = [&](int a, int b) {
        if (depth[a] < depth[b]) swap(a, b);
        int diff = depth[a] - depth[b];
        for (int j = 0; j < lg; ++j) if (diff >> j & 1) a = up[j][a];
        if (a == b) return a;
        for (int j = lg - 1; j >= 0; --j) {
            if (up[j][a] != up[j][b]) {
                a = up[j][a];
                b = up[j][b];
            }
        }
        return parent[a];
    };
    auto treeDistance = [&](int a, int b) {
        int c = lca(a, b);
        return depth[a] + depth[b] - 2 * depth[c];
    };

    vector<int> core;
    core.reserve(2 * n);
    int current = 1;
    vector<int> downward;
    for (int i = 1; i < n; ++i) {
        int next = preorder[i];
        int connector = parent[next];
        int best_distance = treeDistance(current, connector);
        for (int to : graph[next]) {
            if (rank[to] >= i) continue;
            int d = treeDistance(current, to);
            if (d < best_distance) {
                best_distance = d;
                connector = to;
            }
        }
        int join = lca(current, connector);
        int u = current;
        while (u != join) {
            u = parent[u];
            core.push_back(u);
        }
        downward.clear();
        u = connector;
        while (u != join) {
            downward.push_back(u);
            u = parent[u];
        }
        reverse(downward.begin(), downward.end());
        core.insert(core.end(), downward.begin(), downward.end());
        core.push_back(next);
        current = next;
    }

    int shortcut_padding = k - (int)core.size();
    if (triangle_neighbor == -1) shortcut_padding &= ~1;
    bool shortcut_ok = triangle_neighbor == -1 ||
                       (shortcut_padding >= 3 && (shortcut_padding & 1));
    if (shortcut_ok) {
        vector<int> shortcut;
        shortcut.reserve(k);
        int shortcut_even = triangle_neighbor == -1 ? shortcut_padding
                                                     : shortcut_padding - 3;
        for (int used = 0; used < shortcut_even; used += 2) {
            shortcut.push_back(cheap_neighbor);
            shortcut.push_back(1);
        }
        if (triangle_neighbor != -1) {
            shortcut.push_back(cheap_neighbor);
            shortcut.push_back(triangle_neighbor);
            shortcut.push_back(1);
        }
        shortcut.insert(shortcut.end(), core.begin(), core.end());
        if ((int)shortcut.size() < k) {
            int last = shortcut.empty() ? 1 : shortcut.back();
            shortcut.push_back(graph[last][0]);
        }
        int64 shortcut_value = evaluateWalk(shortcut);
        if (shortcut_value > value) {
            value = shortcut_value;
            walk = move(shortcut);
        }
    }
    return {value, move(walk)};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, k;
    if (!(cin >> n >> m >> k)) return 0;
    vector<int64> rate(n + 1);
    for (int v = 1; v <= n; ++v) cin >> rate[v];
    vector<vector<int>> graph(n + 1);
    for (int i = 0; i < m; ++i) {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }

    vector<int> distance(n + 1, -1), bfs_parent(n + 1, 0), bfs_order;
    bfs_order.reserve(n);
    queue<int> q;
    distance[1] = 0;
    bfs_parent[1] = -1;
    q.push(1);
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        bfs_order.push_back(v);
        for (int to : graph[v]) {
            if (distance[to] == -1) {
                distance[to] = distance[v] + 1;
                bfs_parent[to] = v;
                q.push(to);
            }
        }
    }

    int64 total_rate = accumulate(rate.begin() + 1, rate.end(), int64(0));
    TourResult best{-1, {}};
    vector<int> padding_neighbors = graph[1];
    sort(padding_neighbors.begin(), padding_neighbors.end(), [&](int a, int b) {
        if (rate[a] != rate[b]) return rate[a] < rate[b];
        return a < b;
    });
    padding_neighbors.erase(unique(padding_neighbors.begin(), padding_neighbors.end()),
                            padding_neighbors.end());
    vector<int> all_root_neighbors = padding_neighbors;
    if ((int)padding_neighbors.size() > 2) padding_neighbors.resize(2);
    if (padding_neighbors.empty()) padding_neighbors.push_back(-1);

    vector<char> is_root_neighbor(n + 1, false);
    for (int v : all_root_neighbors) is_root_neighbor[v] = true;
    vector<pair<int, int>> triangle_pairs;
    for (int a : all_root_neighbors) {
        for (int b : graph[a]) {
            if (a != b && is_root_neighbor[b]) triangle_pairs.push_back({a, b});
        }
    }
    sort(triangle_pairs.begin(), triangle_pairs.end(), [&](auto x, auto y) {
        if (rate[x.first] != rate[y.first]) return rate[x.first] < rate[y.first];
        if (rate[x.second] != rate[y.second]) return rate[x.second] < rate[y.second];
        return x < y;
    });
    triangle_pairs.erase(unique(triangle_pairs.begin(), triangle_pairs.end()),
                         triangle_pairs.end());
    if ((int)triangle_pairs.size() > 8) triangle_pairs.resize(8);

    // Different ways of resolving equal-distance BFS parents.  The exact
    // objective calculation below chooses the right one for this instance.
    for (int mode : {2}) {
        vector<int> parent = bfs_parent;
        if (mode != 0) {
            for (int v = 2; v <= n; ++v) {
                parent[v] = 0;
                for (int to : graph[v]) {
                    if (distance[to] + 1 != distance[v]) continue;
                    bool take = parent[v] == 0;
                    int old = parent[v];
                    if (!take && mode == 1) { // closest absolute rate
                        int64 a = llabs(rate[to] - rate[v]);
                        int64 b = llabs(rate[old] - rate[v]);
                        take = a < b || (a == b && to < old);
                    } else if (!take && mode == 2) { // closest relative rate
                        i128 a = (i128)llabs(rate[to] - rate[v]);
                        i128 b = (i128)llabs(rate[old] - rate[v]);
                        i128 lhs = a * (rate[old] + rate[v]);
                        i128 rhs = b * (rate[to] + rate[v]);
                        take = lhs < rhs || (lhs == rhs && to < old);
                    } else if (!take && mode == 3) { // polarize around mean
                        bool high = (i128)rate[v] * n >= total_rate;
                        take = high ? (rate[to] > rate[old] ||
                                       (rate[to] == rate[old] && to < old))
                                    : (rate[to] < rate[old] ||
                                       (rate[to] == rate[old] && to < old));
                    } else if (!take && mode == 4) {
                        take = rate[to] < rate[old] ||
                               (rate[to] == rate[old] && to < old);
                    } else if (!take && mode == 5) {
                        take = rate[to] > rate[old] ||
                               (rate[to] == rate[old] && to < old);
                    } else if (!take && mode == 6) {
                        take = to < old;
                    }
                    if (take) parent[v] = to;
                }
            }
        }
        for (int cheap_neighbor : padding_neighbors) {
            TourResult candidate = buildTour(parent, bfs_order, graph, rate, n, k,
                                             cheap_neighbor);
            if (candidate.value > best.value) best = move(candidate);
        }
        for (auto [a, b] : triangle_pairs) {
            TourResult candidate = buildTour(parent, bfs_order, graph, rate, n, k, a, b);
            if (candidate.value > best.value) best = move(candidate);
        }
    }

    // A globally low-rate-first frontier tree is a useful counterpoint to
    // shortest-path trees on sparse instances.
    vector<int> prim_parent(n + 1, 0), prim_order;
    vector<char> in_prim(n + 1, false);
    using FrontierEdge = tuple<int64, int, int>; // rate, vertex, parent
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> frontier;
    prim_parent[1] = -1;
    in_prim[1] = true;
    prim_order.push_back(1);
    for (int to : graph[1]) frontier.push({rate[to], to, 1});
    while (!frontier.empty()) {
        auto [ignored_rate, v, from] = frontier.top();
        frontier.pop();
        if (in_prim[v]) continue;
        in_prim[v] = true;
        prim_parent[v] = from;
        prim_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_prim[to]) frontier.push({rate[to], to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(prim_parent, prim_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(prim_parent, prim_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }

    vector<int> smooth_parent(n + 1, 0), smooth_order;
    vector<char> in_smooth(n + 1, false);
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> smooth_frontier;
    smooth_parent[1] = -1;
    in_smooth[1] = true;
    smooth_order.push_back(1);
    for (int to : graph[1])
        smooth_frontier.push({llabs(rate[to] - rate[1]), to, 1});
    while (!smooth_frontier.empty()) {
        auto [ignored_gap, v, from] = smooth_frontier.top();
        smooth_frontier.pop();
        if (in_smooth[v]) continue;
        in_smooth[v] = true;
        smooth_parent[v] = from;
        smooth_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_smooth[to])
                smooth_frontier.push({llabs(rate[to] - rate[v]), to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(smooth_parent, smooth_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(smooth_parent, smooth_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }

    vector<int> relative_parent(n + 1, 0), relative_order;
    vector<char> in_relative(n + 1, false);
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> relative_frontier;
    auto relativeGap = [&](int a, int b) -> int64 {
        return llabs(rate[a] - rate[b]) * 1000000000LL / (rate[a] + rate[b]);
    };
    relative_parent[1] = -1;
    in_relative[1] = true;
    relative_order.push_back(1);
    for (int to : graph[1]) relative_frontier.push({relativeGap(1, to), to, 1});
    while (!relative_frontier.empty()) {
        auto [ignored_gap, v, from] = relative_frontier.top();
        relative_frontier.pop();
        if (in_relative[v]) continue;
        in_relative[v] = true;
        relative_parent[v] = from;
        relative_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_relative[to])
                relative_frontier.push({relativeGap(v, to), to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(relative_parent, relative_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(relative_parent, relative_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }

    for (int64 bias : {-1000LL, -100LL, -20LL, -15LL, -10LL, -7LL, -5LL,
                       10LL, 100LL, 1000LL}) {
        vector<int> biased_parent(n + 1, 0), biased_order;
        vector<char> in_biased(n + 1, false);
        priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> biased_frontier;
        auto biasedGap = [&](int a, int b) -> int64 {
            return relativeGap(a, b) + bias * rate[b];
        };
        biased_parent[1] = -1;
        in_biased[1] = true;
        biased_order.push_back(1);
        for (int to : graph[1]) biased_frontier.push({biasedGap(1, to), to, 1});
        while (!biased_frontier.empty()) {
            auto [ignored_gap, v, from] = biased_frontier.top();
            biased_frontier.pop();
            if (in_biased[v]) continue;
            in_biased[v] = true;
            biased_parent[v] = from;
            biased_order.push_back(v);
            for (int to : graph[v]) {
                if (!in_biased[to]) biased_frontier.push({biasedGap(v, to), to, v});
            }
        }
        for (int cheap_neighbor : padding_neighbors) {
            TourResult candidate = buildTour(biased_parent, biased_order, graph, rate, n, k,
                                             cheap_neighbor);
            if (candidate.value > best.value) best = move(candidate);
        }
    }

    for (int64 bias : {-100LL, -10LL, -1LL, 1LL, 10LL, 100LL}) {
        vector<int> biased_parent(n + 1, 0), biased_order;
        vector<char> in_biased(n + 1, false);
        priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> biased_frontier;
        auto biasedGap = [&](int a, int b) -> int64 {
            return 10 * llabs(rate[a] - rate[b]) - bias * rate[b];
        };
        biased_parent[1] = -1;
        in_biased[1] = true;
        biased_order.push_back(1);
        for (int to : graph[1]) biased_frontier.push({biasedGap(1, to), to, 1});
        while (!biased_frontier.empty()) {
            auto [ignored_gap, v, from] = biased_frontier.top();
            biased_frontier.pop();
            if (in_biased[v]) continue;
            in_biased[v] = true;
            biased_parent[v] = from;
            biased_order.push_back(v);
            for (int to : graph[v]) {
                if (!in_biased[to]) biased_frontier.push({biasedGap(v, to), to, v});
            }
        }
        for (int cheap_neighbor : padding_neighbors) {
            TourResult candidate = buildTour(biased_parent, biased_order, graph, rate, n, k,
                                             cheap_neighbor);
            if (candidate.value > best.value) best = move(candidate);
        }
    }

    vector<int> middle_parent(n + 1, 0), middle_order;
    vector<char> in_middle(n + 1, false);
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> middle_frontier;
    auto middleGap = [&](int a, int b) -> int64 {
        int64 diff = llabs(rate[a] - rate[b]);
        return diff * diff * 1000000LL / (rate[a] + rate[b]);
    };
    middle_parent[1] = -1;
    in_middle[1] = true;
    middle_order.push_back(1);
    for (int to : graph[1]) middle_frontier.push({middleGap(1, to), to, 1});
    while (!middle_frontier.empty()) {
        auto [ignored_gap, v, from] = middle_frontier.top();
        middle_frontier.pop();
        if (in_middle[v]) continue;
        in_middle[v] = true;
        middle_parent[v] = from;
        middle_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_middle[to]) middle_frontier.push({middleGap(v, to), to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(middle_parent, middle_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(middle_parent, middle_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }

    vector<int> high_parent(n + 1, 0), high_order;
    vector<char> in_high(n + 1, false);
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> high_frontier;
    auto highGap = [&](int a, int b) -> int64 {
        int64 sum = rate[a] + rate[b];
        i128 scaled = (i128)llabs(rate[a] - rate[b]) * 1000000000000000LL;
        return (int64)(scaled / sum / sum);
    };
    high_parent[1] = -1;
    in_high[1] = true;
    high_order.push_back(1);
    for (int to : graph[1]) high_frontier.push({highGap(1, to), to, 1});
    while (!high_frontier.empty()) {
        auto [ignored_gap, v, from] = high_frontier.top();
        high_frontier.pop();
        if (in_high[v]) continue;
        in_high[v] = true;
        high_parent[v] = from;
        high_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_high[to]) high_frontier.push({highGap(v, to), to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(high_parent, high_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(high_parent, high_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }

    vector<int> important_parent(n + 1, 0), important_order;
    vector<char> in_important(n + 1, false);
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> important_frontier;
    auto importantGap = [&](int a, int b) -> int64 {
        return llabs(rate[a] - rate[b]) * (rate[a] + rate[b]);
    };
    important_parent[1] = -1;
    in_important[1] = true;
    important_order.push_back(1);
    for (int to : graph[1]) important_frontier.push({importantGap(1, to), to, 1});
    while (!important_frontier.empty()) {
        auto [ignored_gap, v, from] = important_frontier.top();
        important_frontier.pop();
        if (in_important[v]) continue;
        in_important[v] = true;
        important_parent[v] = from;
        important_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_important[to])
                important_frontier.push({importantGap(v, to), to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(important_parent, important_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(important_parent, important_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }

    vector<int> by_rate(n), rate_band(n + 1);
    iota(by_rate.begin(), by_rate.end(), 1);
    sort(by_rate.begin(), by_rate.end(), [&](int a, int b) {
        if (rate[a] != rate[b]) return rate[a] < rate[b];
        return a < b;
    });
    for (int bands : {2, 3, 4, 5, 6, 8, 10, 12, 16, 24}) {
    for (int i = 0; i < n; ++i) rate_band[by_rate[i]] = bands * i / n;
    vector<int> band_parent(n + 1, 0), band_order;
    vector<char> in_band(n + 1, false);
    priority_queue<FrontierEdge, vector<FrontierEdge>, greater<FrontierEdge>> band_frontier;
    auto bandGap = [&](int a, int b) -> int64 {
        return (int64)abs(rate_band[a] - rate_band[b]) * 1000000000000000LL
               + relativeGap(a, b);
    };
    band_parent[1] = -1;
    in_band[1] = true;
    band_order.push_back(1);
    for (int to : graph[1]) band_frontier.push({bandGap(1, to), to, 1});
    while (!band_frontier.empty()) {
        auto [ignored_gap, v, from] = band_frontier.top();
        band_frontier.pop();
        if (in_band[v]) continue;
        in_band[v] = true;
        band_parent[v] = from;
        band_order.push_back(v);
        for (int to : graph[v]) {
            if (!in_band[to]) band_frontier.push({bandGap(v, to), to, v});
        }
    }
    for (int cheap_neighbor : padding_neighbors) {
        TourResult candidate = buildTour(band_parent, band_order, graph, rate, n, k,
                                         cheap_neighbor);
        if (candidate.value > best.value) best = move(candidate);
    }
    for (auto [a, b] : triangle_pairs) {
        TourResult candidate = buildTour(band_parent, band_order, graph, rate, n, k, a, b);
        if (candidate.value > best.value) best = move(candidate);
    }
    }

    for (int i = 0; i < k; ++i) {
        if (i) cout << ' ';
        cout << best.walk[i];
    }
    cout << '\n';
    return 0;
}
// EVOLVE-BLOCK-END
