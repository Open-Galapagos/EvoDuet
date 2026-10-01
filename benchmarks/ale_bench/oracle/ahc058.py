# EVOLVE-BLOCK-START
CPP_CODE = '''
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using ull = unsigned long long;
using lll = __int128;

static int N = 10, L = 4, T = 500;
static ll K = 1;
static ll A[10];
static ll C[4][10];
static ll Cf[40];                 // Cf[m] = C[m/10][m%10]
static double dA[10];
static double c1[512], c2[512], c3[512], c4[512];

static double TL = 1.85;
static chrono::steady_clock::time_point T0;
static inline double now() { return chrono::duration<double>(chrono::steady_clock::now() - T0).count(); }

static ull rngs = 88172645463325252ULL;
static inline ull rnd() { rngs ^= rngs << 7; rngs ^= rngs >> 9; rngs ^= rngs << 8; return rngs; }
static inline int rndi(int n) { return (int)(rnd() % (ull)n); }
static inline double rndd() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }

struct Sim {
    lll apple, income;
    ll b0[10], b1[10], b2[10];
    int p[40];
    int dyn[10], ndyn;

    void reset() {
        apple = K; income = 0; ndyn = 0;
        for (int j = 0; j < 10; j++) { b0[j] = b1[j] = b2[j] = 1; }
        memset(p, 0, sizeof(p));
    }
    inline lll cost(int m) const { return (lll)Cf[m] * (p[m] + 1); }
    inline void doBuy(int m) {
        apple -= (lll)Cf[m] * (p[m] + 1);
        p[m]++;
        if (m < 10) { income += (lll)A[m] * b0[m]; return; }
        int j = m - (m / 10) * 10;
        for (int k = 0; k < ndyn; k++) if (dyn[k] == j) return;
        dyn[ndyn++] = j;
    }
    inline void step() {
        apple += income;
        for (int k = 0; k < ndyn; k++) {
            int j = dyn[k];
            ll d = b1[j] * p[10 + j];
            if (d) { b0[j] += d; if (p[j]) income += (lll)A[j] * p[j] * d; }
            b1[j] += b2[j] * p[20 + j];
            b2[j] += p[30 + j];
        }
    }
};

static inline double lg(lll v) { return v <= 1 ? 0.0 : log2((double)v); }

// projected gains of one more purchase at each level of id j, r turns left
static inline void gainsFor(const Sim &s, int j, int r, double q, double g[4]) {
    double B0 = (double)s.b0[j], B1 = (double)s.b1[j], B2 = (double)s.b2[j];
    double P0 = s.p[j], P1 = s.p[10 + j], P2 = s.p[20 + j], P3 = s.p[30 + j];
    double Q0 = max(P0, q), Q1 = max(P1, q), Q2 = max(P2, q), Q3 = max(P3, q);
    g[0] = dA[j] * (c1[r] * B0 + c2[r] * B1 * P1 + c3[r] * B2 * P2 * P1 + c4[r] * P3 * P2 * P1);
    g[1] = dA[j] * Q0 * (c2[r] * B1 + c3[r] * B2 * Q2 + c4[r] * Q3 * Q2);
    g[2] = dA[j] * Q0 * Q1 * (c3[r] * B2 + c4[r] * Q3);
    g[3] = dA[j] * Q0 * Q1 * Q2 * c4[r];
}

// one-pass greedy; phase 1 (t<S) uses mask1 and horizon hor1, phase 2 uses mask2
static vector<unsigned char> greedyGen(double lam, double q, int mask1, int mask2, int S, int hor1) {
    Sim s; s.reset();
    vector<unsigned char> seq;
    seq.reserve(T);
    for (int t = 0; t < T; t++) {
        int mask = (t < S) ? mask1 : mask2;
        int r = (t < S) ? min(T - t, hor1 - t) : (T - t);
        if (r < 1) r = 1;
        double bestv = 0; int bestm = -1;
        for (int j = 0; j < N; j++) {
            double g[4];
            gainsFor(s, j, r, q, g);
            bool ch = (mask >> j) & 1;
            for (int i = 0; i < L; i++) {
                if (i >= 1 && !ch) continue;
                int m = i * 10 + j;
                lll cs = s.cost(m);
                if (cs > s.apple) continue;
                double v = g[i] - lam * (double)(ll)cs;
                if (v > bestv) { bestv = v; bestm = m; }
            }
        }
        if (bestm >= 0) { s.doBuy(bestm); seq.push_back((unsigned char)bestm); }
        s.step();
    }
    return seq;
}

// ---- exact decode of a purchase sequence (buy each item as soon as affordable) ----
static lll runSeq(const vector<unsigned char> &seq, vector<signed char> *acts) {
    Sim s; s.reset();
    const unsigned char *sp = seq.data();
    size_t ptr = 0, n = seq.size();
    for (int t = 0; t < T; t++) {
        int a = -1;
        if (ptr < n) {
            int m = sp[ptr];
            if ((lll)Cf[m] * (s.p[m] + 1) <= s.apple) { s.doBuy(m); a = m; ptr++; }
        }
        if (acts) (*acts)[t] = (signed char)a;
        s.step();
    }
    return s.apple;
}

// ---- checkpoint ladder so a mutation only re-simulates the affected suffix ----
static const int CKSTEP = 32, MAXCK = 24;
struct CK { Sim s; int turn; };
static CK cks[MAXCK];
static int nck = 1;

static lll buildCK(const vector<unsigned char> &seq) {
    Sim s; s.reset();
    const unsigned char *sp = seq.data();
    size_t ptr = 0, n = seq.size();
    cks[0].s = s; cks[0].turn = 0; nck = 1;
    for (int t = 0; t < T; t++) {
        if (ptr < n) {
            int m = sp[ptr];
            if ((lll)Cf[m] * (s.p[m] + 1) <= s.apple) { s.doBuy(m); ptr++; }
        }
        s.step();
        if (nck < MAXCK && ptr == (size_t)nck * CKSTEP) { cks[nck].s = s; cks[nck].turn = t + 1; nck++; }
    }
    return s.apple;
}

// rebuild the ladder from checkpoint z onward (prefix below z*CKSTEP is unchanged)
static lll buildCKFrom(const vector<unsigned char> &seq, int z) {
    if (z >= nck) z = nck - 1;
    if (z < 0) z = 0;
    Sim s = cks[z].s;
    int t = cks[z].turn;
    const unsigned char *sp = seq.data();
    size_t ptr = (size_t)z * CKSTEP, n = seq.size();
    nck = z + 1;
    for (; t < T; t++) {
        if (ptr < n) {
            int m = sp[ptr];
            if ((lll)Cf[m] * (s.p[m] + 1) <= s.apple) { s.doBuy(m); ptr++; }
        }
        s.step();
        if (nck < MAXCK && ptr == (size_t)nck * CKSTEP) { cks[nck].s = s; cks[nck].turn = t + 1; nck++; }
    }
    return s.apple;
}

static lll evalFrom(const vector<unsigned char> &seq, int kmin) {
    int z = kmin / CKSTEP;
    if (z >= nck) z = nck - 1;
    Sim s = cks[z].s;
    int t = cks[z].turn;
    const unsigned char *sp = seq.data();
    size_t ptr = (size_t)z * CKSTEP, n = seq.size();
    for (; t < T; t++) {
        if (ptr < n) {
            int m = sp[ptr];
            if ((lll)Cf[m] * (s.p[m] + 1) <= s.apple) { s.doBuy(m); ptr++; }
        }
        s.step();
    }
    return s.apple;
}

static double TS = 0.008, TE = 0.0003;
static int BSZ = 6, PSELF = 60;
static int NSEED = 14; static long long ITC = 0;

static double runLS(vector<unsigned char> &seq, double tStart, double tEnd) {
    vector<unsigned char> cur = seq, cand, best = seq;
    double curL = lg(buildCK(cur));
    double bestL = curL;
    double span = max(1e-6, tEnd - tStart);
    double tn = now();
    while (tn < tEnd) {
        double temp = TS * pow(TE / TS, (tn - tStart) / span);
        double invT = 1.0 / temp;
        for (int rep = 0; rep < 96; rep++) { ITC++;
            cand = cur;
            int n = (int)cand.size();
            int type = rndi(100);
            int kmin, mm;
            if (rndi(100) < PSELF && n > 0) mm = cand[rndi(n)]; else mm = rndi(40);
            if (type < 28 && n > 0) {
                int k = rndi(n); kmin = k;
                if (cand[k] == (unsigned char)mm) continue;
                cand[k] = (unsigned char)mm;
            } else if (type < 42) {
                int k = n ? rndi(n + 1) : 0; kmin = k;
                cand.insert(cand.begin() + k, (unsigned char)mm);
            } else if (type < 56 && n > 1) {
                int k = rndi(n); kmin = k;
                cand.erase(cand.begin() + k);
            } else if (type < 66 && n > 1) {
                int k = rndi(n), d = 1 + rndi(16);
                int k2 = rndi(2) ? k + d : k - d;
                if (k2 < 0) k2 = 0; if (k2 >= n) k2 = n - 1;
                if (k == k2 || cand[k] == cand[k2]) continue;
                kmin = min(k, k2);
                unsigned char m = cand[k];
                cand.erase(cand.begin() + k);
                cand.insert(cand.begin() + k2, m);
            } else if (type < 76) {
                int k = 1 + rndi(BSZ);
                int p = n ? rndi(n + 1) : 0; kmin = p;
                cand.insert(cand.begin() + p, k, (unsigned char)mm);
            } else if (type < 86 && n > 2) {
                int k = 1 + rndi(BSZ);
                int p = rndi(n); kmin = p;
                if (p + k > n) k = n - p;
                cand.erase(cand.begin() + p, cand.begin() + p + k);
            } else if (type < 96 && n > 2) {
                int k = 1 + rndi(BSZ);
                int p = rndi(n);
                if (p + k > n) k = n - p;
                unsigned char blk[64];
                memcpy(blk, cand.data() + p, k);
                cand.erase(cand.begin() + p, cand.begin() + p + k);
                int p2 = rndi((int)cand.size() + 1);
                kmin = min(p, p2);
                cand.insert(cand.begin() + p2, blk, blk + k);
            } else if (n > 0) {
                int j1 = mm - (mm / 10) * 10, j2 = rndi(10);
                if (j1 == j2) continue;
                int lvmask = rndi(2) ? 15 : 14;
                int first = n;
                for (int z = 0; z < n; z++) {
                    int m = cand[z];
                    if (m - (m / 10) * 10 == j1 && ((lvmask >> (m / 10)) & 1)) {
                        cand[z] = (unsigned char)((m / 10) * 10 + j2);
                        if (z < first) first = z;
                    }
                }
                if (first == n) continue;
                kmin = first;
            } else continue;
            if ((int)cand.size() > T) cand.resize(T);
            int zmin = min(kmin / CKSTEP, nck - 1);
            double vl = lg(evalFrom(cand, kmin));
            if (vl >= curL) {
                curL = vl; cur.swap(cand); buildCKFrom(cur, zmin);
                if (vl > bestL) { bestL = vl; best = cur; }
            } else if (rndd() < exp((vl - curL) * invT)) { curL = vl; cur.swap(cand); buildCKFrom(cur, zmin); }
        }
        tn = now();
    }
    seq = best;
    return bestL;
}

int main(int argc, char **argv) {
    T0 = chrono::steady_clock::now();
    if (argc > 1) { double tl = atof(argv[1]); if (tl > 0.2) TL = tl - 0.15; }
    if (const char *e = getenv("AHC_TL")) TL = atof(e);
    if (const char *e = getenv("AHC_TS")) TS = atof(e);
    if (const char *e = getenv("AHC_TE")) TE = atof(e);
    if (const char *e = getenv("AHC_NSEED")) NSEED = atoi(e);
    if (const char *e = getenv("AHC_BSZ")) BSZ = atoi(e);
    if (const char *e = getenv("AHC_PSELF")) PSELF = atoi(e);
    if (scanf("%d %d %d %lld", &N, &L, &T, &K) != 4) return 0;
    for (int j = 0; j < N; j++) { if (scanf("%lld", &A[j]) != 1) return 0; dA[j] = (double)A[j]; }
    for (int i = 0; i < L; i++) for (int j = 0; j < N; j++) { if (scanf("%lld", &C[i][j]) != 1) return 0; Cf[i * 10 + j] = C[i][j]; }
    for (int r = 0; r <= T + 1 && r < 512; r++) {
        double R = r;
        c1[r] = R; c2[r] = R * (R - 1) / 2; c3[r] = c2[r] * (R - 2) / 3; c4[r] = c3[r] * (R - 3) / 4;
        if (c2[r] < 0) c2[r] = 0;
        if (c3[r] < 0) c3[r] = 0;
        if (c4[r] < 0) c4[r] = 0;
    }

    struct Seed { vector<unsigned char> seq; double val; };
    vector<Seed> seeds;
    int full = (1 << N) - 1;
    {
        static const double lams[] = {1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 100.0};
        static const double qs[] = {0.0, 1.0, 8.0, 40.0};
        vector<int> masks;
        masks.push_back(full);
        for (int j = 0; j < N; j++) masks.push_back(1 << j);
        for (int j = 1; j < N; j++) masks.push_back(1 | (1 << j));
        for (int mask : masks) {
            double bv = -1; vector<unsigned char> bs;
            for (double lam : lams) for (double q : qs) {
                vector<unsigned char> sq = greedyGen(lam, q, mask, mask, 0, T);
                double v = lg(runSeq(sq, nullptr));
                if (v > bv) { bv = v; bs = sq; }
            }
            if (bv > 0) seeds.push_back({bs, bv});
        }
        static const int Ss[] = {40, 80, 130, 190};
        static const double lam2[] = {1.0, 4.0, 20.0};
        static const double q2[] = {0.0, 8.0};
        vector<int> cheap(N); iota(cheap.begin(), cheap.end(), 0);
        sort(cheap.begin(), cheap.end(), [](int a, int b) {
            return log((double)C[1][a]) + log((double)C[2][a]) + log((double)C[3][a])
                 < log((double)C[1][b]) + log((double)C[2][b]) + log((double)C[3][b]); });
        vector<int> m1s; m1s.push_back(full);
        for (int z = 0; z < 2 && z < (int)cheap.size(); z++) m1s.push_back(1 << cheap[z]);
        for (int j = 0; j < N; j++) for (int m1 : m1s) {
            double bv = -1; vector<unsigned char> bs;
            for (int S : Ss) for (double lam : lam2) for (double q : q2) for (int hv = 0; hv < 2; hv++) {
                vector<unsigned char> sq = greedyGen(lam, q, m1, 1 << j, S, hv ? T : S);
                double v = lg(runSeq(sq, nullptr));
                if (v > bv) { bv = v; bs = sq; }
            }
            if (bv > 0) seeds.push_back({bs, bv});
        }
    }
    sort(seeds.begin(), seeds.end(), [](const Seed &a, const Seed &b) { return a.val > b.val; });
    {
        vector<Seed> u;
        for (auto &sd : seeds) {
            bool dup = false;
            for (auto &t : u) if (fabs(t.val - sd.val) < 1e-9) { dup = true; break; }
            if (!dup) u.push_back(sd);
        }
        seeds.swap(u);
    }
    if (seeds.empty()) seeds.push_back({greedyGen(1.0, 0.0, full, full, 0, T), 0.0});

    double remain = TL - now();
    int M = (int)seeds.size();
    if (M > NSEED) { seeds.resize(NSEED); M = NSEED; }
    vector<double> frac = {0.28, 0.24, 0.22, 0.26};
    vector<int> keep = {M, max(1, M / 3), max(1, M / 6), 1};
    double tcur = now();
    for (int ph = 0; ph < (int)frac.size(); ph++) {
        int m = min((int)seeds.size(), keep[ph]);
        if (m <= 0) break;
        double per = remain * frac[ph] / m;
        for (int z = 0; z < m; z++) {
            double te = min(TL, tcur + per);
            seeds[z].val = runLS(seeds[z].seq, tcur, te);
            tcur = now();
        }
        sort(seeds.begin(), seeds.end(), [](const Seed &a, const Seed &b) { return a.val > b.val; });
        if (tcur >= TL) break;
    }

    vector<signed char> acts(T, -1);
    lll fin = runSeq(seeds[0].seq, &acts);
    string out; out.reserve(T * 5);
    char buf[16];
    for (int t = 0; t < T; t++) {
        if (acts[t] < 0) out += "-1\\n";
        else { int m = acts[t]; snprintf(buf, sizeof(buf), "%d %d\\n", m / 10, m % 10); out += buf; }
    }
    fputs(out.c_str(), stdout);
    fprintf(stderr, "seeds=%d score=%.4f its=%lld\\n", M, lg(fin), ITC);
    return 0;
}
'''
# EVOLVE-BLOCK-END
