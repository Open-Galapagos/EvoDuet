// EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

using ll = long long;

struct FastRng {
    uint64_t x;
    explicit FastRng(uint64_t seed) : x(seed) {}
    uint64_t next() {
        x ^= x << 7;
        x ^= x >> 9;
        return x;
    }
    int operator()(int n) { return int(next() % uint64_t(n)); }
};

static ll costOf(const vector<vector<int>>& g, const vector<unsigned char>& initial,
                 const vector<int>& order) {
    const int n = (int)g.size();
    vector<unsigned char> col = initial, dead(n, 0);
    ll black = 0, cost = 0;
    for (unsigned char x : col) black += x;
    for (int u : order) {
        cost += black;
        black -= col[u];
        dead[u] = 1;
        for (int v : g[u]) if (!dead[v]) {
            if (col[v]) --black;
            else ++black;
            col[v] ^= 1;
        }
    }
    return cost;
}

static vector<int> bfsOrder(const vector<vector<int>>& g, int root) {
    int n = (int)g.size();
    vector<int> q(n), order;
    vector<unsigned char> seen(n, 0);
    order.reserve(n);
    int head = 0, tail = 0;
    q[tail++] = root;
    seen[root] = 1;
    while (head < tail) {
        int u = q[head++];
        order.push_back(u);
        for (int v : g[u]) if (!seen[v]) {
            seen[v] = 1;
            q[tail++] = v;
        }
    }
    return order;
}

static vector<int> dfsOrder(const vector<vector<int>>& g, int root, bool post) {
    int n = (int)g.size();
    vector<int> parent(n, -2), order;
    order.reserve(n);
    vector<int> st;
    st.reserve(n);
    st.push_back(root);
    parent[root] = -1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        order.push_back(u);
        for (int i = (int)g[u].size() - 1; i >= 0; --i) {
            int v = g[u][i];
            if (v == parent[u]) continue;
            parent[v] = u;
            st.push_back(v);
        }
    }
    if (post) reverse(order.begin(), order.end());
    return order;
}

// Greedy restricted to leaves.  Different tie modes deliberately produce
// rather different roots and sibling orders.
static vector<int> leafGreedy(const vector<vector<int>>& g,
                              const vector<unsigned char>& initial, int tieMode) {
    const int n = (int)g.size();
    vector<int> deg(n), ver(n, 0), order;
    vector<unsigned char> col = initial, dead(n, 0);
    order.reserve(n);
    for (int i = 0; i < n; ++i) deg[i] = (int)g[i].size();

    using Item = tuple<int,int,int,int>; // marginal, tie, vertex, version
    priority_queue<Item, vector<Item>, greater<Item>> pq;
    auto loneNeighbor = [&](int u) {
        for (int v : g[u]) if (!dead[v]) return v;
        return -1;
    };
    auto pushLeaf = [&](int u) {
        if (dead[u] || deg[u] > 1) return;
        int v = loneNeighbor(u);
        int d = -int(col[u]) + (v < 0 ? 0 : 1 - 2 * int(col[v]));
        int tie;
        if (tieMode == 0) tie = u;
        else if (tieMode == 1) tie = -u;
        else if (tieMode == 2) tie = -(int)g[u].size();
        else tie = (int)((uint32_t)u * 2654435761u);
        pq.emplace(d, tie, u, ver[u]);
    };
    for (int i = 0; i < n; ++i) if (deg[i] <= 1) pushLeaf(i);

    while ((int)order.size() < n) {
        int u = -1;
        while (!pq.empty()) {
            auto [oldD, tie, x, oldVer] = pq.top(); pq.pop();
            if (dead[x] || deg[x] > 1 || oldVer != ver[x]) continue;
            int v = loneNeighbor(x);
            int d = -int(col[x]) + (v < 0 ? 0 : 1 - 2 * int(col[v]));
            if (d != oldD) {
                ++ver[x];
                pushLeaf(x);
                continue;
            }
            u = x;
            break;
        }
        if (u < 0) { // Only possible for the final isolated vertex after stale entries.
            for (int i = 0; i < n; ++i) if (!dead[i]) { u = i; break; }
        }
        dead[u] = 1;
        order.push_back(u);
        for (int v : g[u]) if (!dead[v]) {
            col[v] ^= 1;
            --deg[v];
            ++ver[v];
            if (deg[v] <= 1) pushLeaf(v);
            // If v was already a leaf, its other neighbor's color did not
            // change.  Other existing leaves adjacent to v cannot occur in a tree
            // unless the remaining component has at most two vertices.
            if (deg[v] == 1) {
                int w = loneNeighbor(v);
                if (w >= 0 && deg[w] <= 1) { ++ver[w]; pushLeaf(w); }
            }
        }
    }
    return order;
}

// Approximate global marginal heap, repaired exactly at extraction.  Updates
// through low-degree flipped vertices are propagated; high-degree neighborhood
// additions remain lazy, avoiding quadratic behavior on stars.
static vector<int> marginalGreedy(const vector<vector<int>>& g,
                                  const vector<unsigned char>& initial,
                                  uint64_t seed, int samples, int propLimit) {
    const int n = (int)g.size();
    vector<unsigned char> col = initial, dead(n, 0);
    vector<int> deg(n), estimate(n), ver(n, 0), alive(n), pos(n), order;
    order.reserve(n);
    for (int u = 0; u < n; ++u) {
        deg[u] = (int)g[u].size();
        int d = -int(col[u]);
        for (int v : g[u]) d += 1 - 2 * int(col[v]);
        estimate[u] = d;
        alive[u] = pos[u] = u;
    }
    using Item = tuple<int,int,int>; // estimate, random tie, vertex
    priority_queue<Item, vector<Item>, greater<Item>> pq;
    FastRng rng(seed);
    auto push = [&](int u) {
        ++ver[u];
        pq.emplace(estimate[u], int(rng.next() >> 33), u);
    };
    // Version is intentionally not stored in Item: duplicate entries with the
    // same current estimate are harmless.  This saves memory in the hot heap.
    for (int u = 0; u < n; ++u) pq.emplace(estimate[u], int(rng.next() >> 33), u);

    auto exactMarginal = [&](int u) {
        int d = -int(col[u]);
        for (int v : g[u]) if (!dead[v]) d += 1 - 2 * int(col[v]);
        return d;
    };
    int aliveCount = n;
    while (aliveCount) {
        int best = -1, bestD = INT_MAX;
        // Repair several heap heads.  Limiting repair work also protects
        // adversarial high-degree instances.
        for (int tries = 0; tries < 24 && !pq.empty(); ++tries) {
            auto [key, tie, u] = pq.top(); pq.pop();
            if (dead[u] || key != estimate[u]) continue;
            int d = exactMarginal(u);
            if (d != estimate[u]) {
                estimate[u] = d;
                pq.emplace(d, int(rng.next() >> 33), u);
                continue;
            }
            best = u; bestD = d;
            pq.emplace(key, tie, u);
            break;
        }
        // Uniform probes recover vertices whose true marginal improved while
        // buried under a stale high-degree neighborhood update.
        int probes = min(samples, aliveCount);
        for (int k = 0; k < probes; ++k) {
            int u = alive[rng(aliveCount)];
            int d = exactMarginal(u);
            if (d < bestD || (d == bestD && (rng.next() & 1))) {
                bestD = d;
                best = u;
            }
        }
        if (best < 0) best = alive[aliveCount - 1];

        int x = best;
        dead[x] = 1;
        order.push_back(x);
        int ix = pos[x], z = alive[aliveCount - 1];
        alive[ix] = z; pos[z] = ix;
        --aliveCount;

        int cx = col[x];
        for (int v : g[x]) if (!dead[v]) {
            int old = col[v];
            // Remove x's old contribution and account for v's own flip.
            estimate[v] -= 1 - 2 * cx;
            estimate[v] += 2 * old - 1;
            col[v] ^= 1;
            --deg[v];
            push(v);
            if ((int)g[v].size() <= propLimit) {
                int delta = 2 * (2 * old - 1);
                for (int w : g[v]) if (!dead[w]) {
                    estimate[w] += delta;
                    push(w);
                }
            }
        }
    }
    return order;
}

// One left-to-right adjacent-exchange pass.  Before a pair (x,y), both
// alternatives reach exactly the same state after two removals.  Therefore
// y,x is strictly better iff marginal(y) < marginal(x) in the current state.
static vector<int> adjacentSweep(const vector<vector<int>>& g,
                                 const vector<unsigned char>& initial,
                                 vector<int> order, long long scanBudget) {
    int n = (int)g.size();
    vector<unsigned char> col = initial, dead(n, 0);
    auto marginal = [&](int u) {
        int d = -int(col[u]);
        for (int v : g[u]) if (!dead[v]) d += 1 - 2 * int(col[v]);
        return d;
    };
    auto remove = [&](int u) {
        dead[u] = 1;
        for (int v : g[u]) if (!dead[v]) col[v] ^= 1;
    };
    for (int i = 0; i + 1 < n; ++i) {
        int x = order[i], y = order[i + 1];
        long long need = g[x].size() + g[y].size();
        if (need <= scanBudget) {
            scanBudget -= need;
            if (marginal(y) < marginal(x)) {
                swap(order[i], order[i + 1]);
                x = y;
            }
        }
        remove(x);
    }
    return order;
}

// Exact minimum prefix-set chain for genuinely small instances.
static vector<int> exactSubset(const vector<vector<int>>& g,
                               const vector<unsigned char>& initial) {
    int n = (int)g.size(), states = 1 << n;
    vector<uint32_t> adj(n, 0);
    for (int u = 0; u < n; ++u)
        for (int v : g[u]) adj[u] |= 1u << v;
    vector<unsigned char> f(states), take(states, 0);
    vector<int> dp(states, INT_MAX / 4);
    int f0 = 0;
    for (int u = 0; u < n; ++u) f0 += initial[u];
    f[0] = (unsigned char)f0;
    dp[0] = f0;
    for (int mask = 1; mask < states; ++mask) {
        int u = __builtin_ctz((unsigned)mask);
        int prev = mask ^ (1 << u);
        int cu = int(initial[u]) ^ (__builtin_parity((unsigned)prev & adj[u]));
        int delta = -cu;
        for (int v : g[u]) if (!(prev & (1 << v))) {
            int cv = int(initial[v]) ^ (__builtin_parity((unsigned)prev & adj[v]));
            delta += 1 - 2 * cv;
        }
        f[mask] = (unsigned char)(int(f[prev]) + delta);

        int bits = mask;
        while (bits) {
            int b = __builtin_ctz((unsigned)bits);
            int cand = dp[mask ^ (1 << b)] + int(f[mask]);
            if (cand < dp[mask]) {
                dp[mask] = cand;
                take[mask] = (unsigned char)b;
            }
            bits &= bits - 1;
        }
    }
    vector<int> order(n);
    int mask = states - 1;
    for (int p = n - 1; p >= 0; --p) {
        int u = take[mask];
        order[p] = u;
        mask ^= 1 << u;
    }
    return order;
}

// Rebuild an order while allowing the best current marginal among a short
// prefix of the remaining base order to jump forward.
static vector<int> windowReorder(const vector<vector<int>>& g,
                                 const vector<unsigned char>& initial,
                                 const vector<int>& base, int width,
                                 long long scanBudget) {
    int n = (int)g.size(), next = 0;
    vector<unsigned char> col = initial, dead(n, 0);
    vector<int> window, out;
    window.reserve(width + 1);
    out.reserve(n);
    while (next < n && (int)window.size() < width) window.push_back(base[next++]);
    while (!window.empty()) {
        int at = 0, bestD = INT_MAX;
        bool exhausted = false;
        for (int i = 0; i < (int)window.size(); ++i) {
            int u = window[i];
            if ((long long)g[u].size() > scanBudget) { exhausted = true; break; }
            scanBudget -= g[u].size();
            int d = -int(col[u]);
            for (int v : g[u]) if (!dead[v]) d += 1 - 2 * int(col[v]);
            if (d < bestD) { bestD = d; at = i; }
        }
        if (exhausted && scanBudget <= 0) at = 0;
        int u = window[at];
        window.erase(window.begin() + at);
        if (next < n) window.push_back(base[next++]);
        out.push_back(u);
        dead[u] = 1;
        for (int v : g[u]) if (!dead[v]) col[v] ^= 1;
    }
    return out;
}

static vector<int> windowLookahead(const vector<vector<int>>& g,
                                   const vector<unsigned char>& initial,
                                   const vector<int>& base, int width,
                                   int firstWeight, long long scanBudget) {
    int n = (int)g.size(), next = 0;
    vector<unsigned char> col = initial, dead(n, 0);
    vector<int> parent(n, -2), st(1, 0);
    parent[0] = -1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        for (int v : g[u]) if (v != parent[u]) {
            parent[v] = u;
            st.push_back(v);
        }
    }
    vector<int> window, out, d;
    window.reserve(width + 1); out.reserve(n); d.reserve(width);
    while (next < n && (int)window.size() < width) window.push_back(base[next++]);
    while (!window.empty()) {
        int k = (int)window.size();
        long long need = 0;
        for (int u : window) need += g[u].size();
        // Exact marginals scan the window neighborhoods; all pair effects are
        // constant-time parent/grandparent tests on a tree.
        long long work = need + 1LL * k * k;
        if (work > scanBudget) {
            // Finish with the cheaper one-step rule once the explicit budget is
            // depleted; preserve the remaining base order.
            vector<int> tail = window;
            while (next < n) tail.push_back(base[next++]);
            out.insert(out.end(), tail.begin(), tail.end());
            return out;
        }
        scanBudget -= work;
        d.assign(k, 0);
        for (int i = 0; i < k; ++i) {
            int u = window[i], x = -int(col[u]);
            for (int v : g[u]) if (!dead[v]) x += 1 - 2 * int(col[v]);
            d[i] = x;
        }
        int at = 0, bestScore = INT_MAX;
        for (int i = 0; i < k; ++i) {
            int u = window[i];
            int nextD = 0;
            if (k > 1) {
                nextD = INT_MAX;
                for (int j = 0; j < k; ++j) if (j != i) {
                    int v = window[j], after = d[j];
                    if (parent[u] == v || parent[v] == u) {
                        after += 2 * int(col[v]) - 1;
                        after -= 1 - 2 * int(col[u]);
                    } else {
                        int w = -1;
                        if (parent[u] >= 0 && parent[u] == parent[v]) w = parent[u];
                        else if (parent[u] >= 0 && parent[parent[u]] == v) w = parent[u];
                        else if (parent[v] >= 0 && parent[parent[v]] == u) w = parent[v];
                        if (w >= 0 && !dead[w])
                            after += 2 * (2 * int(col[w]) - 1);
                    }
                    nextD = min(nextD, after);
                }
            }
            int score = firstWeight * d[i] + nextD;
            if (score < bestScore) { bestScore = score; at = i; }
        }
        int u = window[at];
        window.erase(window.begin() + at);
        if (next < n) window.push_back(base[next++]);
        out.push_back(u);
        dead[u] = 1;
        for (int v : g[u]) if (!dead[v]) col[v] ^= 1;
    }
    return out;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<int>> g(n);
    for (int i = 1, a, b; i < n; ++i) {
        cin >> a >> b;
        --a; --b;
        g[a].push_back(b);
        g[b].push_back(a);
    }
    string s;
    cin >> s;
    vector<unsigned char> initial(n);
    for (int i = 0; i < n; ++i) initial[i] = (s[i] == 'B');

    if (n <= 21) {
        vector<int> answer = exactSubset(g, initial);
        for (int i = 0; i < n; ++i) cout << answer[i] + 1 << (i + 1 == n ? '\n' : ' ');
        return 0;
    }

    int maxDeg = 0, blackMaxDeg = -1;
    for (int i = 1; i < n; ++i) if (g[i].size() > g[maxDeg].size()) maxDeg = i;
    for (int i = 0; i < n; ++i) if (initial[i] &&
        (blackMaxDeg < 0 || g[i].size() > g[blackMaxDeg].size())) blackMaxDeg = i;
    if (blackMaxDeg < 0) blackMaxDeg = maxDeg;

    vector<int> best = bfsOrder(g, 0); // exact checker reference; never regress.
    ll bestCost = costOf(g, initial, best);
    auto consider = [&](vector<int>&& cand) {
        ll c = costOf(g, initial, cand);
        if (c < bestCost) { bestCost = c; best = move(cand); }
    };

    if (n > 1) {
        consider(bfsOrder(g, maxDeg));
        consider(bfsOrder(g, blackMaxDeg));
        auto x = bfsOrder(g, 0); reverse(x.begin(), x.end()); consider(move(x));
        x = bfsOrder(g, maxDeg); reverse(x.begin(), x.end()); consider(move(x));
        consider(dfsOrder(g, 0, false));
        consider(dfsOrder(g, 0, true));
        consider(dfsOrder(g, maxDeg, true));
    }

    for (int mode = 0; mode < 3; ++mode) consider(leafGreedy(g, initial, mode));

    vector<int> staticOrder(n);
    iota(staticOrder.begin(), staticOrder.end(), 0);
    sort(staticOrder.begin(), staticOrder.end(), [&](int a, int b) {
        if (initial[a] != initial[b]) return initial[a] > initial[b];
        if (initial[a]) return g[a].size() > g[b].size();
        return g[a].size() < g[b].size();
    });
    consider(move(staticOrder));

    consider(marginalGreedy(g, initial, 0x9e3779b97f4a7c15ULL, 10, 10));
    consider(marginalGreedy(g, initial, 0xd1b54a32d192ed03ULL, 18, 6));
    consider(marginalGreedy(g, initial, 0x94d049bb133111ebULL, 32, 14));
    consider(marginalGreedy(g, initial, 0x8538ec4a92ea2b6dULL, 48, 4));

    vector<int> beforeWindow = best;
    consider(windowReorder(g, initial, beforeWindow, 256, 530LL * n));
    vector<int> afterWindow = best;
    consider(windowReorder(g, initial, afterWindow, 32, 76LL * n));
    vector<int> beforeLookahead = best;
    consider(windowLookahead(g, initial, beforeLookahead, 48, 2, 2420LL * n));

    for (int pass = 0; pass < 6; ++pass) {
        vector<int> improved = adjacentSweep(g, initial, best, 24LL * n);
        ll c = costOf(g, initial, improved);
        if (c >= bestCost) break;
        bestCost = c;
        best = move(improved);
    }

    for (int i = 0; i < n; ++i) {
        if (i) cout << ' ';
        cout << best[i] + 1;
    }
    cout << '\n';
    return 0;
}
// EVOLVE-BLOCK-END
