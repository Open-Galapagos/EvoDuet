// EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

using ll = long long;

struct SplitMix64 {
    static uint64_t mix(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    size_t operator()(ll x) const {
        static const uint64_t seed = chrono::steady_clock::now().time_since_epoch().count();
        return mix((uint64_t)x + seed);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, L, a, b;
    ll c;
    if (!(cin >> n >> L >> a >> b >> c)) return 0;

    vector<ll> x(n + 1), y(n + 1), w(n + 1), u(n + 1), v(n + 1);
    unordered_map<ll, set<pair<ll,int>>, SplitMix64> byU, byV;
    byU.reserve(n * 2 + 1);
    byV.reserve(n * 2 + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> x[i] >> y[i] >> w[i];
        u[i] = x[i] + y[i];
        v[i] = x[i] - y[i];
        byU[u[i]].insert({v[i], i});
        byV[v[i]].insert({u[i], i});
    }

    const ll D = llabs(x[a] - x[b]) + llabs(y[a] - y[b]);
    vector<int> parent(n + 1, -1), order;
    vector<vector<int>> child(n + 1);
    queue<int> q;

    auto erase_point = [&](int id) {
        auto iu = byU.find(u[id]);
        if (iu != byU.end()) iu->second.erase({v[id], id});
        auto iv = byV.find(v[id]);
        if (iv != byV.end()) iv->second.erase({u[id], id});
    };

    parent[a] = 0;
    parent[b] = a;
    erase_point(a);
    erase_point(b);
    q.push(a);
    q.push(b);
    order.push_back(a);
    order.push_back(b);

    auto discover_u = [&](ll key, ll lo, ll hi, int p) {
        auto mi = byU.find(key);
        if (mi == byU.end()) return;
        auto &s = mi->second;
        while (true) {
            auto it = s.lower_bound({lo, numeric_limits<int>::min()});
            if (it == s.end() || it->first > hi) break;
            int z = it->second;
            parent[z] = p;
            child[p].push_back(z);
            order.push_back(z);
            q.push(z);
            erase_point(z);
        }
    };
    auto discover_v = [&](ll key, ll lo, ll hi, int p) {
        auto mi = byV.find(key);
        if (mi == byV.end()) return;
        auto &s = mi->second;
        while (true) {
            auto it = s.lower_bound({lo, numeric_limits<int>::min()});
            if (it == s.end() || it->first > hi) break;
            int z = it->second;
            parent[z] = p;
            child[p].push_back(z);
            order.push_back(z);
            q.push(z);
            erase_point(z);
        }
    };

    while (!q.empty()) {
        int p = q.front(); q.pop();
        discover_u(u[p] - D, v[p] - D, v[p] + D, p);
        discover_u(u[p] + D, v[p] - D, v[p] + D, p);
        discover_v(v[p] - D, u[p] - D, u[p] + D, p);
        discover_v(v[p] + D, u[p] - D, u[p] + D, p);
    }

    // F[z]: best gain from descendants of z, returning to its incoming edge.
    // H[z]: best gain from descendants of z when allowed to finish anywhere.
    vector<ll> F(n + 1, 0), H(n + 1, 0);
    vector<int> cntF(n + 1, 0), cntH(n + 1, 0), finalChild(n + 1, -1);
    vector<char> useF(n + 1, 0);

    for (int oi = (int)order.size() - 1; oi >= 0; --oi) {
        int z = order[oi];
        ll sum = 0;
        int moves = 0;
        for (int t : child[z]) {
            ll g = w[t] - c + F[t];
            if (g > 0) {
                sum += g;
                moves += 1 + cntF[t];
            }
        }
        if (sum > c) {
            F[z] = sum - c;
            cntF[z] = moves + 1;
            useF[z] = 1;
        }

        ll bestAdv = 0;
        int fc = -1;
        for (int t : child[z]) {
            ll g = w[t] - c + F[t];
            ll nonret = w[t] - c + H[t];
            ll adv = nonret - max(0LL, g);
            if (adv > bestAdv) {
                bestAdv = adv;
                fc = t;
            }
        }
        H[z] = sum + bestAdv;
        cntH[z] = moves;
        if (fc != -1) {
            ll g = w[fc] - c + F[fc];
            if (g > 0) cntH[z] -= 1 + cntF[fc];
            cntH[z] += 1 + cntH[fc];
            finalChild[z] = fc;
        }
    }

    vector<pair<int,int>> seq;
    seq.reserve(min(L + 100, 400000));

    auto density_better = [&](int s, int t) {
        ll gs = w[s] - c + F[s], gt = w[t] - c + F[t];
        ll ms = 1 + cntF[s], mt = 1 + cntF[t];
        __int128 lhs = (__int128)gs * mt;
        __int128 rhs = (__int128)gt * ms;
        if (lhs != rhs) return lhs > rhs;
        return gs > gt;
    };

    // Iterative generators avoid recursion depth issues. mode 0 = returning F,
    // mode 1 = non-returning H. Each frame is entered with edge {p,z} current.
    struct Frame { int p, z, mode, phase, idx, fc; vector<int> kids; };
    auto generate = [&](int p0, int z0, int mode0) {
        vector<Frame> st;
        auto make_frame = [&](int p, int z, int mode) {
            Frame f{p, z, mode, 0, 0, mode ? finalChild[z] : -1, {}};
            if (mode == 0 && !useF[z]) return f;
            for (int t : child[z]) {
                ll g = w[t] - c + F[t];
                if (g > 0 && t != f.fc) f.kids.push_back(t);
            }
            sort(f.kids.begin(), f.kids.end(), density_better);
            return f;
        };
        st.push_back(make_frame(p0, z0, mode0));
        while (!st.empty() && (int)seq.size() <= L + 5) {
            Frame &f = st.back();
            if (f.idx < (int)f.kids.size()) {
                int t = f.kids[f.idx++];
                seq.push_back({f.z, t});
                st.push_back(make_frame(f.z, t, 0));
                continue;
            }
            if (f.mode == 1 && f.phase == 0 && f.fc != -1) {
                f.phase = 1;
                int t = f.fc;
                seq.push_back({f.z, t});
                st.push_back(make_frame(f.z, t, 1));
                continue;
            }
            int p = f.p, z = f.z, mode = f.mode;
            st.pop_back();
            if (mode == 0 && useF[z]) seq.push_back({p, z});
        }
    };

    ll optionAB = F[a] + H[b];
    ll optionBA = F[b] + H[a];
    ll optionA = H[a], optionB = H[b];
    int choice = 0;
    ll bestRoot = 0;
    auto take_choice = [&](ll gain, int ch) {
        if (gain > bestRoot) { bestRoot = gain; choice = ch; }
    };
    take_choice(optionA, 1);
    take_choice(optionB, 2);
    take_choice(optionAB, 3);
    take_choice(optionBA, 4);
    if (choice == 1) generate(b, a, 1);
    else if (choice == 2) generate(a, b, 1);
    else if (choice == 3) { generate(b, a, 0); generate(a, b, 1); }
    else if (choice == 4) { generate(a, b, 0); generate(b, a, 1); }

    vector<pair<int,int>> treeSeq = move(seq);

    // A second construction works in the real graph rather than only in its BFS
    // tree. It normally gains one fresh point per move, and uses its own touched
    // spanning tree only when it has to relocate to an older frontier vertex.
    for (int z : order) if (z != a && z != b) {
        byU[u[z]].insert({v[z], z});
        byV[v[z]].insert({u[z], z});
    }

    auto candidate = [&](int z) {
        int ans = -1;
        auto consider_u = [&](ll key, ll lo, ll hi) {
            auto mi = byU.find(key);
            if (mi == byU.end()) return;
            auto it = mi->second.lower_bound({lo, numeric_limits<int>::min()});
            if (it != mi->second.end() && it->first <= hi) {
                int t = it->second;
                if (ans == -1 || w[t] > w[ans]) ans = t;
            }
        };
        auto consider_v = [&](ll key, ll lo, ll hi) {
            auto mi = byV.find(key);
            if (mi == byV.end()) return;
            auto it = mi->second.lower_bound({lo, numeric_limits<int>::min()});
            if (it != mi->second.end() && it->first <= hi) {
                int t = it->second;
                if (ans == -1 || w[t] > w[ans]) ans = t;
            }
        };
        consider_u(u[z] - D, v[z] - D, v[z] + D);
        consider_u(u[z] + D, v[z] - D, v[z] + D);
        consider_v(v[z] - D, u[z] - D, u[z] + D);
        consider_v(v[z] + D, u[z] - D, u[z] + D);
        return ans;
    };

    static constexpr int LOG = 19;
    vector<array<int,LOG>> up(n + 1);
    vector<int> tp(n + 1, -1), dep(n + 1, 0), frontier;
    tp[a] = 0;
    tp[b] = a;
    dep[b] = 1;
    up[b][0] = a;
    for (int k = 1; k < LOG; ++k) up[b][k] = up[up[b][k-1]][k-1];
    frontier.push_back(a);
    frontier.push_back(b);

    auto lca = [&](int s, int t) {
        if (dep[s] < dep[t]) swap(s, t);
        int dif = dep[s] - dep[t];
        for (int k = 0; k < LOG; ++k) if (dif >> k & 1) s = up[s][k];
        if (s == t) return s;
        for (int k = LOG - 1; k >= 0; --k) if (up[s][k] != up[t][k]) {
            s = up[s][k]; t = up[t][k];
        }
        return tp[s];
    };
    auto tdist = [&](int s, int t) {
        int z = lca(s, t);
        return dep[s] + dep[t] - 2 * dep[z];
    };

    vector<pair<int,int>> greedySeq;
    greedySeq.reserve(L + 1);
    int curEdge = b; // a non-root node represents its incoming touched-tree edge
    size_t frontierPos = 0;
    vector<char> exhausted(n + 1, 0);

    auto relocate = [&](int targetEdge) {
        if (curEdge == targetEdge) return;
        int ce[2] = {curEdge, tp[curEdge]};
        int te[2] = {targetEdge, tp[targetEdge]};
        int bs = ce[0], bt = te[0], bd = numeric_limits<int>::max();
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            int dd = tdist(ce[i], te[j]);
            if (dd < bd) { bd = dd; bs = ce[i]; bt = te[j]; }
        }
        int z = lca(bs, bt);
        vector<int> edges, down;
        for (int s = bs; s != z; s = tp[s]) edges.push_back(s);
        for (int t = bt; t != z; t = tp[t]) down.push_back(t);
        reverse(down.begin(), down.end());
        edges.insert(edges.end(), down.begin(), down.end());
        edges.push_back(targetEdge);
        int prev = curEdge;
        for (int e : edges) if (e != prev) {
            greedySeq.push_back({tp[e], e});
            prev = e;
            if ((int)greedySeq.size() >= L) break;
        }
        curEdge = prev;
    };

    while ((int)greedySeq.size() < L) {
        int p = curEdge, qv = tp[curEdge];
        int rp = candidate(p), rq = candidate(qv);
        int pivot = p, fresh = rp;
        if (fresh == -1 || (rq != -1 && w[rq] > w[fresh])) {
            pivot = qv; fresh = rq;
        }
        if (fresh == -1) {
            int target = -1;
            while (frontierPos < frontier.size()) {
                int z = frontier[frontierPos];
                if (!exhausted[z] && candidate(z) != -1) { target = z; break; }
                exhausted[z] = 1;
                ++frontierPos;
            }
            if (target == -1) break;
            int targetEdge = (target == a ? b : target);
            relocate(targetEdge);
            if ((int)greedySeq.size() >= L) break;
            continue;
        }
        greedySeq.push_back({pivot, fresh});
        erase_point(fresh);
        tp[fresh] = pivot;
        dep[fresh] = dep[pivot] + 1;
        up[fresh][0] = pivot;
        for (int k = 1; k < LOG; ++k) up[fresh][k] = up[up[fresh][k-1]][k-1];
        frontier.push_back(fresh);
        curEdge = fresh;
    }

    // Any prefix is valid. Compare both constructions by their actual best
    // prefix profit, rather than trusting the planning estimates.
    auto assess = [&](const vector<pair<int,int>> &s) {
        vector<char> touched(n + 1, 0);
        touched[a] = touched[b] = 1;
        ll profit = w[a] + w[b], bestProfit = profit;
        int bestLen = 0, lim = min(L, (int)s.size());
        for (int i = 0; i < lim; ++i) {
            profit -= c;
            int p = s[i].first, qv = s[i].second;
            if (!touched[p]) { touched[p] = 1; profit += w[p]; }
            if (!touched[qv]) { touched[qv] = 1; profit += w[qv]; }
            if (profit > bestProfit) { bestProfit = profit; bestLen = i + 1; }
        }
        return pair<ll,int>{bestProfit, bestLen};
    };

    vector<pair<int,int>> spiderSeq;
    pair<ll,int> spiderScore = {w[a] + w[b], 0};
    vector<int> deg(n + 1, 0);
    for (int z : order) if (z != a) {
        ++deg[z];
        ++deg[parent[z]];
    }
    int center = -1, branchVertices = 0;
    for (int z : order) if (deg[z] > 2) {
        center = z;
        ++branchVertices;
    }
    if (branchVertices == 0) center = a;
    if (branchVertices <= 1 && (center == a || center == b)) {
        vector<vector<int>> tadj(n + 1);
        for (int z : order) if (z != a) {
            tadj[z].push_back(parent[z]);
            tadj[parent[z]].push_back(z);
        }
        int initialNeighbor = (center == a ? b : a);
        vector<vector<int>> arms;
        int initialArm = -1;
        for (int nb : tadj[center]) {
            vector<int> arm;
            int prv = center, cur = nb;
            while (true) {
                arm.push_back(cur);
                int nxt = -1;
                for (int z : tadj[cur]) if (z != prv) { nxt = z; break; }
                if (nxt == -1) break;
                prv = cur; cur = nxt;
            }
            if (nb == initialNeighbor) initialArm = (int)arms.size();
            arms.push_back(move(arm));
        }

        struct Marginal { int arm, pos, cost; ll value; };
        struct MargCmp {
            bool operator()(const Marginal &s, const Marginal &t) const {
                __int128 lhs = (__int128)s.value * t.cost;
                __int128 rhs = (__int128)t.value * s.cost;
                if (lhs != rhs) return lhs < rhs;
                return s.value < t.value;
            }
        };

        int m = (int)arms.size();
        auto test_take = [&](const vector<int> &take, int finish) {
            vector<pair<int,int>> cand;
            cand.reserve(L + 2);
            bool currentInitial = true;
            auto add_initial_return = [&](int t) {
                for (int j = 1; j <= t; ++j)
                    cand.push_back({arms[initialArm][j-1], arms[initialArm][j]});
                for (int j = t - 1; j >= 0; --j) {
                    int p = (j == 0 ? center : arms[initialArm][j-1]);
                    cand.push_back({p, arms[initialArm][j]});
                }
            };
            if (initialArm != -1 && initialArm != finish && take[initialArm] > 0)
                add_initial_return(take[initialArm]);

            vector<int> ret;
            for (int i = 0; i < m; ++i)
                if (i != finish && i != initialArm && take[i] > 0) ret.push_back(i);
            sort(ret.begin(), ret.end(), [&](int s, int t) {
                ll vs = 0, vt = 0;
                for (int j = 0; j < take[s]; ++j) vs += w[arms[s][j]];
                for (int j = 0; j < take[t]; ++j) vt += w[arms[t][j]];
                return (__int128)vs * (2 * take[t] - 1) >
                       (__int128)vt * (2 * take[s] - 1);
            });
            for (int i : ret) {
                int t = take[i];
                cand.push_back({center, arms[i][0]});
                for (int j = 1; j < t; ++j) cand.push_back({arms[i][j-1], arms[i][j]});
                for (int j = t - 2; j >= 0; --j) {
                    int p = (j == 0 ? center : arms[i][j-1]);
                    cand.push_back({p, arms[i][j]});
                }
                currentInitial = false;
            }

            int ft = take[finish];
            if (ft > 0) {
                if (finish == initialArm) {
                    if (!currentInitial) cand.push_back({center, initialNeighbor});
                    for (int j = 1; j <= ft; ++j)
                        cand.push_back({arms[finish][j-1], arms[finish][j]});
                } else {
                    cand.push_back({center, arms[finish][0]});
                    for (int j = 1; j < ft; ++j)
                        cand.push_back({arms[finish][j-1], arms[finish][j]});
                }
            }
            auto sc = assess(cand);
            if (sc.first > spiderScore.first) {
                spiderScore = sc;
                spiderSeq = move(cand);
            }
        };

        for (int finish = 0; finish < m; ++finish) {
            vector<int> take(m, 0);
            priority_queue<Marginal, vector<Marginal>, MargCmp> pq;
            auto avail = [&](int i) {
                return (int)arms[i].size() - (i == initialArm ? 1 : 0);
            };
            auto point_at = [&](int i, int pos) {
                return arms[i][pos + (i == initialArm ? 1 : 0)];
            };
            auto marginal_cost = [&](int i, int pos) {
                if (i == finish) return 1;
                if (i == initialArm) return 2;
                return pos == 0 ? 1 : 2;
            };
            auto push_next = [&](int i) {
                int pos = take[i];
                if (pos >= avail(i)) return;
                int mc = marginal_cost(i, pos);
                int z = point_at(i, pos);
                pq.push({i, pos, mc, w[z] - c * mc});
            };
            for (int i = 0; i < m; ++i) push_next(i);
            int used = 0;
            while (!pq.empty()) {
                Marginal e = pq.top(); pq.pop();
                if (e.pos != take[e.arm]) continue;
                if (used + e.cost > L) continue;
                used += e.cost;
                ++take[e.arm];
                push_next(e.arm);
            }
            test_take(take, finish);

            // Pool adjacent increasing marginal densities. A low-value point
            // immediately before a rich stretch should be judged together with
            // the stretch it unlocks, rather than permanently blocking it.
            struct Block { int cnt, cost; ll value; };
            vector<vector<Block>> blocks(m);
            for (int i = 0; i < m; ++i) {
                for (int pos = 0; pos < avail(i); ++pos) {
                    int mc = marginal_cost(i, pos);
                    int z = point_at(i, pos);
                    blocks[i].push_back({1, mc, w[z] - c * mc});
                    auto &bs = blocks[i];
                    while (bs.size() >= 2) {
                        Block &p = bs[bs.size()-2], &qv = bs.back();
                        if ((__int128)p.value * qv.cost >= (__int128)qv.value * p.cost) break;
                        Block merged{p.cnt + qv.cnt, p.cost + qv.cost, p.value + qv.value};
                        bs.pop_back(); bs.back() = merged;
                    }
                }
            }
            struct Pick { int arm, bi, cnt, cost; ll value; };
            auto pickCmp = [](const Pick &s, const Pick &t) {
                __int128 lhs = (__int128)s.value * t.cost;
                __int128 rhs = (__int128)t.value * s.cost;
                if (lhs != rhs) return lhs < rhs;
                return s.value < t.value;
            };
            priority_queue<Pick, vector<Pick>, decltype(pickCmp)> hp(pickCmp);
            for (int i = 0; i < m; ++i) if (!blocks[i].empty()) {
                Block e = blocks[i][0];
                hp.push({i, 0, e.cnt, e.cost, e.value});
            }
            vector<int> take2(m, 0);
            int used2 = 0;
            while (!hp.empty()) {
                Pick e = hp.top(); hp.pop();
                if (used2 + e.cost <= L) {
                    used2 += e.cost;
                    take2[e.arm] += e.cnt;
                    int nb = e.bi + 1;
                    if (nb < (int)blocks[e.arm].size()) {
                        Block qv = blocks[e.arm][nb];
                        hp.push({e.arm, nb, qv.cnt, qv.cost, qv.value});
                    }
                } else {
                    int end = take2[e.arm] + e.cnt;
                    while (take2[e.arm] < end) {
                        int mc = marginal_cost(e.arm, take2[e.arm]);
                        if (used2 + mc > L) break;
                        used2 += mc;
                        ++take2[e.arm];
                    }
                }
            }
            test_take(take2, finish);

            auto coordinate_improve = [&](vector<int> cur) {
                vector<vector<ll>> pref(m);
                for (int i = 0; i < m; ++i) {
                    pref[i].assign(avail(i) + 1, 0);
                    for (int j = 0; j < avail(i); ++j)
                        pref[i][j+1] = pref[i][j] + w[point_at(i, j)];
                }
                auto take_cost = [&](int i, int t) {
                    if (i == finish) return t;
                    if (i == initialArm) return 2 * t;
                    return t == 0 ? 0 : 2 * t - 1;
                };
                auto max_take = [&](int i, int budget) {
                    if (budget < 0) return 0;
                    if (i == finish) return min(avail(i), budget);
                    if (i == initialArm) return min(avail(i), budget / 2);
                    if (budget == 0) return 0;
                    return min(avail(i), (budget + 1) / 2);
                };
                int totalCost = 0;
                for (int i = 0; i < m; ++i) totalCost += take_cost(i, cur[i]);
                for (int round = 0; round < 3; ++round) {
                    bool changed = false;
                    for (int i = 0; i < m; ++i) for (int j = i + 1; j < m; ++j) {
                        int other = totalCost - take_cost(i, cur[i]) - take_cost(j, cur[j]);
                        int budget = L - other;
                        ll bestVal = pref[i][cur[i]] + pref[j][cur[j]]
                                   - c * 1LL * (take_cost(i, cur[i]) + take_cost(j, cur[j]));
                        int bi = cur[i], bj = cur[j];
                        int limi = max_take(i, budget);
                        for (int ti = 0; ti <= limi; ++ti) {
                            int ci = take_cost(i, ti);
                            int tj = max_take(j, budget - ci);
                            ll val = pref[i][ti] + pref[j][tj]
                                   - c * 1LL * (ci + take_cost(j, tj));
                            if (val > bestVal) { bestVal = val; bi = ti; bj = tj; }
                        }
                        if (bi != cur[i] || bj != cur[j]) {
                            totalCost = other + take_cost(i, bi) + take_cost(j, bj);
                            cur[i] = bi; cur[j] = bj;
                            changed = true;
                        }
                    }
                    if (!changed) break;
                }
                test_take(cur, finish);
            };
            coordinate_improve(take);
            coordinate_improve(take2);
            for (int seed = 0; seed < 4; ++seed) {
                vector<int> rndTake(m, 0);
                uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (uint64_t)(finish * 17 + seed + 1);
                int rndCost = 0;
                while (true) {
                    vector<int> can;
                    for (int i = 0; i < m; ++i) if (rndTake[i] < avail(i)) {
                        int mc = marginal_cost(i, rndTake[i]);
                        if (rndCost + mc <= L) can.push_back(i);
                    }
                    if (can.empty()) break;
                    rng ^= rng << 7; rng ^= rng >> 9; rng ^= rng << 8;
                    int i = can[rng % can.size()];
                    rndCost += marginal_cost(i, rndTake[i]);
                    ++rndTake[i];
                }
                coordinate_improve(rndTake);
            }
        }
    }

    auto at = assess(treeSeq), ag = assess(greedySeq);
    const vector<pair<int,int>> *answer = &treeSeq;
    pair<ll,int> best = at;
    if (ag.first > best.first) { answer = &greedySeq; best = ag; }
    if (spiderScore.first > best.first) { answer = &spiderSeq; best = spiderScore; }

    cout << best.second << '\n';
    for (int i = 0; i < best.second; ++i)
        cout << (*answer)[i].first << ' ' << (*answer)[i].second << '\n';
    return 0;
}
// EVOLVE-BLOCK-END
