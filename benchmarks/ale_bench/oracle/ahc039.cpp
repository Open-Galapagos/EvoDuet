# EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const double TL = 1.90;

static inline double nowsec(){
    using namespace std::chrono;
    static const steady_clock::time_point st = steady_clock::now();
    return duration_cast<duration<double>>(steady_clock::now()-st).count();
}

struct XS{
    uint64_t s;
    XS(uint64_t seed=88172645463325252ULL):s(seed){}
    inline void seed(uint64_t v){ s = v*0x9E3779B97F4A7C15ULL + 12345ULL; if(!s) s=1; for(int i=0;i<8;i++) nx(); }
    inline uint64_t nx(){ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
    inline uint32_t n32(){ return (uint32_t)(nx()>>32); }
    inline int ri(int n){ return (int)(((uint64_t)n32()*(uint64_t)n) >> 32); }
    inline double rd(){ return (nx()>>11)*(1.0/9007199254740992.0); }
};
static XS rng;

static bool canAdd[256], canRem[256];
static int8_t dCadd[256], dCrem[256];
static int8_t n4tab[256];

static void buildTables(){
    for(int m=0;m<256;m++){
        auto bit=[&](int i){ return (m>>i)&1; };
        int runs=0;
        for(int i=0;i<8;i++) if(bit(i) && !bit((i+7)&7)) runs++;
        bool ca = (m!=255) && (runs==1);
        bool cr = ca;
        for(int d=1;d<8;d+=2){
            int a=d-1, b=(d+1)&7;
            if(bit(d) && !bit(a) && !bit(b)) ca=false;
            if(!bit(d) && bit(a) && bit(b)) cr=false;
        }
        canAdd[m]=ca; canRem[m]=cr;
        int n4=0; for(int i=0;i<8;i+=2) n4+=bit(i);
        n4tab[m]=(int8_t)n4;
        int blocks[4][3] = {{0,1,2},{2,3,4},{4,5,6},{6,7,0}};
        int da=0, dr=0;
        for(int k=0;k<4;k++){
            int cnt = bit(blocks[k][0])+bit(blocks[k][1])+bit(blocks[k][2]);
            da += ((cnt+0)&1)? -1 : 1;
            dr += ((cnt+1)&1)? -1 : 1;
        }
        dCadd[m]=(int8_t)da; dCrem[m]=(int8_t)dr;
    }
}

static int N;
static vector<int> fxv, fyv;

static int G, ST, CS;
static vector<int> wt;
static vector<uint8_t> inS;
static vector<int> fr, frpos;
static ll curW; static int curP, curC, curCnt;
static int maxP, maxC;
static int O[8];

static inline int mask8(int p){
    return  (inS[p+O[0]])    | (inS[p+O[1]]<<1) | (inS[p+O[2]]<<2) | (inS[p+O[3]]<<3)
        | (inS[p+O[4]]<<4) | (inS[p+O[5]]<<5) | (inS[p+O[6]]<<6) | (inS[p+O[7]]<<7);
}
static inline bool isCand(int p){
    int i = p % ST, j = p / ST;
    if(i<1||i>G||j<1||j>G) return false;
    int n = inS[p+1]+inS[p-1]+inS[p+ST]+inS[p-ST];
    if(inS[p]) return n<4;
    return n>0;
}
static inline void frAdd(int p){ if(frpos[p]<0){ frpos[p]=(int)fr.size(); fr.push_back(p);} }
static inline void frDel(int p){ int i=frpos[p]; if(i>=0){ int last=fr.back(); fr[i]=last; frpos[last]=i; fr.pop_back(); frpos[p]=-1; } }
static inline void frUpd(int p){ if(isCand(p)) frAdd(p); else frDel(p); }

static void recompute(){
    curW=0; curP=0; curC=0; curCnt=0;
    for(int j=1;j<=G;j++) for(int i=1;i<=G;i++){
        int p=j*ST+i;
        if(inS[p]){
            curCnt++; curW += wt[p];
            curP += 4-(inS[p+1]+inS[p-1]+inS[p+ST]+inS[p-ST]);
        }
    }
    for(int vy=0; vy<=G; vy++) for(int vx=0; vx<=G; vx++){
        int c = inS[vy*ST+vx]+inS[vy*ST+vx+1]+inS[(vy+1)*ST+vx]+inS[(vy+1)*ST+vx+1];
        if(c&1) curC++;
    }
    fr.clear(); frpos.assign((size_t)ST*ST,-1);
    for(int j=1;j<=G;j++) for(int i=1;i<=G;i++){ int p=j*ST+i; if(isCand(p)) frAdd(p); }
}

static void initLevel(int cs){
    CS=cs; G=100000/cs; ST=G+2;
    O[0]=1; O[1]=1+ST; O[2]=ST; O[3]=ST-1; O[4]=-1; O[5]=-1-ST; O[6]=-ST; O[7]=1-ST;
    wt.assign((size_t)ST*ST, 0);
    for(int i=0;i<2*N;i++){
        int ix = fxv[i]/CS; if(ix>=G) ix=G-1;
        int iy = fyv[i]/CS; if(iy>=G) iy=G-1;
        wt[(iy+1)*ST + ix+1] += (i<N ? 1 : -1);
    }
    inS.assign((size_t)ST*ST,(uint8_t)0);
    maxP = 400000/CS; maxC = 1000;
}

// SA on the current level. Returns best W found; leaves inS = best state.
static double muC=0.0, muUp=1.05, muAdd=0.01, muDn=0.99, cTgtF=0.99;
static ll runSA(double tE, double T0, double T1, double& lam, double adaptUp, double adaptDn,
                int adaptEvery, double pTgtF, vector<uint8_t>& bestS){
    recompute();
    ll bW = curW; bestS = inS;
    double tStart = nowsec();
    if(tE <= tStart+0.002) tE = tStart+0.002;
    double T = T0, invT = 1.0/T;
    ll iter=0;
    double pTgt = maxP*pTgtF;
    double cTgt = maxC*cTgtF;
    while(true){
        if((iter & (ll)(adaptEvery-1))==0){
            double t = nowsec();
            if(t >= tE) break;
            double r = (t-tStart)/(tE-tStart);
            T = T0*pow(T1/T0, r); invT = 1.0/T;
            if(curP > pTgt) lam *= adaptUp; else lam *= adaptDn;
            if(lam < 0.0002*CS) lam = 0.0002*CS;
            if(lam > 0.5*CS) lam = 0.5*CS;
            if(curC > cTgt) muC = muC*muUp + muAdd; else muC *= muDn;
            if(muC > 50.0) muC = 50.0;
        }
        iter++;
        if(fr.empty()) break;
        int p = fr[rng.ri((int)fr.size())];
        int m = mask8(p);
        if(inS[p]){
            if(!canRem[m] || curCnt<=1) continue;
            int dP = -4 + 2*n4tab[m];
            if(curP+dP > maxP) continue;
            int dC = dCrem[m];
            if(curC+dC > maxC) continue;
            double dS = -(double)wt[p] - lam*(double)dP - muC*(double)dC;
            if(dS < 0 && rng.rd() >= exp(dS*invT)) continue;
            inS[p]=0; curW -= wt[p]; curP += dP; curC += dC; curCnt--;
        } else {
            if(!canAdd[m]) continue;
            int dP = 4 - 2*n4tab[m];
            if(curP+dP > maxP) continue;
            int dC = dCadd[m];
            if(curC+dC > maxC) continue;
            double dS = (double)wt[p] - lam*(double)dP - muC*(double)dC;
            if(dS < 0 && rng.rd() >= exp(dS*invT)) continue;
            inS[p]=1; curW += wt[p]; curP += dP; curC += dC; curCnt++;
        }
        frUpd(p); frUpd(p+1); frUpd(p-1); frUpd(p+ST); frUpd(p-ST);
        if(curW > bW){ bW=curW; bestS=inS; }
    }
    inS = bestS;
    return bW;
}

int main(){
    nowsec();
    buildTables();
    {
        static char buf[1<<21];
        size_t len = fread(buf,1,sizeof(buf)-1,stdin); buf[len]=0;
        char* p=buf;
        auto readInt=[&]()->int{
            while(*p && (*p<'0'||*p>'9') && *p!='-') p++;
            int sg=1; if(*p=='-'){sg=-1;p++;}
            int v=0; while(*p>='0'&&*p<='9'){ v=v*10+(*p-'0'); p++; }
            return v*sg;
        };
        N=readInt();
        fxv.resize(2*N); fyv.resize(2*N);
        for(int i=0;i<2*N;i++){ fxv[i]=readInt(); fyv[i]=readInt(); }
    }
    auto envd=[&](const char*,double d)->double{ return d; };
    auto envi=[&](const char*,int d)->int{ return d; };

    int NL = envi("NLV",5);
    int allCS[6] = {2000,1000,500,250,125,100};
    double tFrac[6] = {envd("TF0",0.40), envd("TF1",0.52), envd("TF2",0.66), envd("TF3",0.82), 1.0, 1.0};
    double T0base = envd("T0B",1.5), T1base = envd("T1B",0.10);
    double lam0 = envd("LAM0",8.0);
    double adaptUp = envd("AUP",1.02), adaptDn = envd("ADN",0.99);
    int adaptEvery = envi("AEV",256);
    double pTgtF = envd("PTG",0.995);
    muUp = envd("MUP",1.05); muAdd = envd("MADD",0.0); muDn = envd("MDN",0.99); cTgtF = envd("CTG",0.99);
    double POLT = envd("POLT",0.06);
    double TEND = TL - envd("TMARG",0.03) - POLT;
    double HARDEND = TL - envd("TMARG",0.03);
    int R0 = envi("R0",80);
    double T0mul0 = envd("T0MUL0",2.0);

    int BK[6] = {envi("BK0",8), envi("BK1",4), envi("BK2",2), envi("BK3",1), 1, 1};

    vector<vector<uint8_t>> beam;
    vector<ll> beamW;
    vector<double> beamLam;

    auto pushBeam=[&](vector<uint8_t>& st, ll w, double lm, int K){
        beam.push_back(st); beamW.push_back(w); beamLam.push_back(lm);
        // keep top-K
        if((int)beam.size() > K){
            int worst=0; for(int i=1;i<(int)beam.size();i++) if(beamW[i]<beamW[worst]) worst=i;
            beam.erase(beam.begin()+worst); beamW.erase(beamW.begin()+worst); beamLam.erase(beamLam.begin()+worst);
        }
    };

    // ---- level 0 with restarts ----
    initLevel(allCS[0]);
    {
        vector<pair<int,int>> pos;
        for(int j=1;j<=G;j++) for(int i=1;i<=G;i++){ int p=j*ST+i; if(wt[p]>0) pos.push_back({wt[p],p}); }
        sort(pos.rbegin(), pos.rend());
        if(pos.empty()) pos.push_back({0,(G/2)*ST+G/2});
        double t0End = TEND*tFrac[0];
        double sc = (double)CS/500.0; sc*=sc;
        for(int r=0; r<R0; r++){
            double tnow = nowsec();
            double tSlice = tnow + (t0End - tnow)/(double)(R0-r);
            rng.seed(0x1234567ULL + 7919ULL*r);
            fill(inS.begin(), inS.end(), (uint8_t)0);
            int idx = (r==0)?0: rng.ri((int)min<size_t>(pos.size(), 30));
            inS[pos[idx].second]=1;
            double lam = lam0;
            vector<uint8_t> bs;
            ll w = runSA(tSlice, T0base*sc*T0mul0, T1base*sc, lam, adaptUp, adaptDn, adaptEvery, pTgtF, bs);
            pushBeam(bs, w, lam, BK[0]);
            if(nowsec() >= t0End) break;
        }
    }

    // ---- refinement levels with beam ----
    for(int L=1; L<NL; L++){
        int prevCS = CS, prevG = G, prevST = ST;
        int newCS = allCS[L];
        int rr = prevCS/newCS;
        int newG = 100000/newCS, newST = newG+2;
        vector<vector<uint8_t>> ups;
        for(auto& st : beam){
            vector<uint8_t> nS((size_t)newST*newST, 0);
            for(int j=0;j<prevG;j++) for(int i=0;i<prevG;i++)
                if(st[(size_t)(j+1)*prevST+i+1])
                    for(int dy=0;dy<rr;dy++) for(int dx=0;dx<rr;dx++)
                        nS[(size_t)(rr*j+dy+1)*newST + (rr*i+dx+1)] = 1;
            ups.push_back(std::move(nS));
        }
        vector<double> lams = beamLam;
        for(auto& v : lams) v /= (double)rr;
        int nb = (int)ups.size();
        initLevel(newCS);
        double sc = (double)CS/500.0; sc*=sc;
        double tE = (L==NL-1)? TEND : TEND*tFrac[L];
        beam.clear(); beamW.clear(); beamLam.clear();
        int K = BK[L];
        for(int b=0; b<nb; b++){
            double tnow = nowsec();
            double slice = (nb-b>0)? tnow + (tE-tnow)/(double)(nb-b) : tE;
            inS = ups[b];
            double lm = lams[b];
            vector<uint8_t> bs;
            ll w = runSA(slice, T0base*sc, T1base*sc, lm, adaptUp, adaptDn, adaptEvery, pTgtF, bs);
            pushBeam(bs, w, lm, K);
            if(nowsec() >= tE) break;
        }
        if(beam.empty()){ beam.push_back(ups[0]); beamW.push_back(0); beamLam.push_back(lams[0]); }
        if(false) fprintf(stderr,"L%d CS=%d nb=%d bestW=%lld t=%.3f\n", L, CS, nb, *max_element(beamW.begin(),beamW.end()), nowsec());
    }
    double gLam = 1.0;
    {
        int bi=0; for(int i=1;i<(int)beam.size();i++) if(beamW[i]>beamW[bi]) bi=i;
        inS = beam[bi]; gLam = beamLam[bi];
    }
    recompute();

    // ---------- trace boundary ----------
    auto cellIn=[&](int i,int j)->int{
        if(i<0||j<0||i>=G||j>=G) return 0;
        return inS[(j+1)*ST+(i+1)];
    };
    auto edgeExists=[&](int i,int j,int d)->bool{
        if(d==0) return i<G && cellIn(i,j-1)!=cellIn(i,j);
        if(d==2) return i>0 && cellIn(i-1,j-1)!=cellIn(i-1,j);
        if(d==1) return j<G && cellIn(i-1,j)!=cellIn(i,j);
        return j>0 && cellIn(i-1,j-1)!=cellIn(i,j-1);
    };
    int sx=-1, sy=-1, sdir=-1;
    for(int j=0;j<=G && sx<0;j++) for(int i=0;i<=G;i++){
        for(int d=0;d<4;d++) if(edgeExists(i,j,d)){ sx=i; sy=j; sdir=d; break; }
        if(sx>=0) break;
    }
    vector<pair<int,int>> path;
    if(sx>=0){
        int ci=sx, cj=sy, cd=sdir;
        const int DX[4]={1,0,-1,0}, DY[4]={0,1,0,-1};
        do{
            path.push_back({ci,cj});
            ci += DX[cd]; cj += DY[cd];
            int rev = (cd+2)&3, nd=-1;
            for(int k=0;k<4;k++){ if(k==rev) continue; if(edgeExists(ci,cj,k)){ nd=k; break; } }
            if(nd<0) break;
            cd=nd;
            if((int)path.size() > 5*G*G) break;
        } while(!(ci==sx && cj==sy));
    }
    vector<pair<int,int>> out;
    {
        int n=(int)path.size();
        for(int i=0;i<n;i++){
            auto& a=path[(i-1+n)%n]; auto& b=path[i]; auto& c=path[(i+1)%n];
            if((a.first==b.first && b.first==c.first) || (a.second==b.second && b.second==c.second)) continue;
            out.push_back(b);
        }
    }
    vector<ll> VX, VY;
    if((int)out.size()>=4 && (int)out.size()<=1000){
        for(auto&v:out){ VX.push_back((ll)v.first*CS); VY.push_back((ll)v.second*CS); }
    } else {
        VX = {0,1,1,0}; VY = {0,0,1,1};
    }

    // ---------- validity check (mirrors official tester) ----------
    auto validPoly=[&](const vector<ll>& X, const vector<ll>& Y)->bool{
        int m=(int)X.size();
        if(m<4||m>1000) return false;
        ll len=0;
        for(int i=0;i<m;i++){
            ll px=X[i], py=Y[i], qx=X[(i+1)%m], qy=Y[(i+1)%m], rx=X[(i+2)%m], ry=Y[(i+2)%m];
            if(px<0||py<0||px>100000||py>100000) return false;
            if(px==qx&&py==qy) return false;
            if(px==qx){ len += llabs(py-qy); if(qx==rx && (py-qy)*(ry-qy)>0) return false; }
            else if(py==qy){ len += llabs(px-qx); if(qy==ry && (px-qx)*(rx-qx)>0) return false; }
            else return false;
        }
        if(len > 400000) return false;
        for(int i=0;i<m;i++){
            ll p1x=X[i],p1y=Y[i],p2x=X[(i+1)%m],p2y=Y[(i+1)%m];
            ll ax=min(p1x,p2x), bx=max(p1x,p2x), ay=min(p1y,p2y), by=max(p1y,p2y);
            for(int k=2;k<m-1;k++){
                int j=(i+k)%m;
                ll q1x=X[j],q1y=Y[j],q2x=X[(j+1)%m],q2y=Y[(j+1)%m];
                if(max(ax,min(q1x,q2x)) <= min(bx,max(q1x,q2x)) && max(ay,min(q1y,q2y)) <= min(by,max(q1y,q2y))) return false;
            }
        }
        return true;
    };

    vector<ll> BX=VX, BY=VY;   // fallback copy
    // ---------- exact sub-grid edge polish ----------
    if((int)VX.size()>=4 && validPoly(VX,VY) && envi("POL",1)){
        int m=(int)VX.size();
        // orientation: make CCW (positive shoelace)
        ll area2=0;
        for(int i=0;i<m;i++){ int j=(i+1)%m; area2 += VX[i]*VY[j] - VX[j]*VY[i]; }
        if(area2 < 0){ reverse(VX.begin(),VX.end()); reverse(VY.begin(),VY.end()); }
        vector<ll> OX=VX, OY=VY;
        // fish sorted by x and by y
        static vector<int> ax_x, ax_y, ay_x, ay_y, startX, startY;
        static vector<int8_t> ax_w, ay_w;
        {
            int n2=2*N;
            vector<int> cnt(100002,0);
            for(int i=0;i<n2;i++) cnt[fxv[i]]++;
            startX.assign(100002,0);
            for(int x=0;x<100001;x++) startX[x+1]=startX[x]+cnt[x];
            ax_x.assign(n2,0); ax_y.assign(n2,0); ax_w.assign(n2,0);
            vector<int> pos = startX;
            for(int i=0;i<n2;i++){ int k=pos[fxv[i]]++; ax_x[k]=fxv[i]; ax_y[k]=fyv[i]; ax_w[k]=(i<N?1:-1); }
            fill(cnt.begin(),cnt.end(),0);
            for(int i=0;i<n2;i++) cnt[fyv[i]]++;
            startY.assign(100002,0);
            for(int y=0;y<100001;y++) startY[y+1]=startY[y]+cnt[y];
            ay_x.assign(n2,0); ay_y.assign(n2,0); ay_w.assign(n2,0);
            pos = startY;
            for(int i=0;i<n2;i++){ int k=pos[fyv[i]]++; ay_x[k]=fxv[i]; ay_y[k]=fyv[i]; ay_w[k]=(i<N?1:-1); }
        }
        int RNG_ = envi("PRAD", CS/2 - 1);
        double lamLen = gLam/(double)CS;
        if(lamLen < 0) lamLen = 0;
        ll P = 0; for(int i=0;i<m;i++){ int j=(i+1)%m; P += llabs(VX[i]-VX[j]) + llabs(VY[i]-VY[j]); }
        const ll PMAX=400000;
        vector<int> dw(2*RNG_+4);
        int passes=0;
        double lamDecay = envd("PLD",0.6);
        double pFull = envd("PFULL",0.999);
        int stall=0;
        while(nowsec() < HARDEND-0.005){
            bool any=false;
            for(int i=0;i<m;i++){
                if((i&15)==0 && nowsec() >= HARDEND-0.005) break;
                int i1=(i+1)%m, im=(i-1+m)%m, i2=(i+2)%m;
                bool vert = (VX[i]==VX[i1]);
                ll X = vert? VX[i] : VY[i];
                ll o  = vert? OX[i] : OY[i];
                ll ca = vert? VX[im]: VY[im];
                ll cb = vert? VX[i2]: VY[i2];
                ll s1 = vert? min(VY[i],VY[i1]) : min(VX[i],VX[i1]);
                ll s2 = vert? max(VY[i],VY[i1]) : max(VX[i],VX[i1]);
                ll lo = max(0LL, o-RNG_), hi = min(100000LL, o+RNG_);
                if(hi<=lo) continue;
                int W_ = (int)(hi-lo+1);
                for(int k=0;k<W_+2;k++) dw[k]=0;
                if(vert){
                    int a=startX[lo], b=startX[hi+1];
                    for(int k=a;k<b;k++){ int yy=ax_y[k]; if(yy>=s1&&yy<=s2) dw[ax_x[k]-lo] += ax_w[k]; }
                } else {
                    int a=startY[lo], b=startY[hi+1];
                    for(int k=a;k<b;k++){ int xx=ay_x[k]; if(xx>=s1&&xx<=s2) dw[ay_y[k]-lo] += ay_w[k]; }
                }
                // prefix: pre[j] = sum of dw[0..j]
                for(int k=1;k<W_;k++) dw[k]+=dw[k-1];
                // direction in which region grows with increasing coordinate
                int grow;
                if(vert) grow = (VY[i1]>VY[i]) ? +1 : -1;   // CCW: going up -> interior left(x<X) -> +x grows
                else     grow = (VX[i1]>VX[i]) ? -1 : +1;   // going right -> interior above -> -y grows
                auto Fp=[&](ll t)->int{ // prefix sum of strip weights with coord <= t
                    if(t<lo) return 0; if(t>hi) return dw[W_-1];
                    return dw[t-lo];
                };
                ll bestT=X; double bestObj=0.0;
                ll baseP = llabs(X-ca)+llabs(X-cb);
                int FX = (grow>0)? Fp(X) : Fp(X-1);
                for(ll t=lo;t<=hi;t++){
                    if(t==X) continue;
                    ll d1=llabs(t-ca), d2=llabs(t-cb);
                    if(d1<1||d2<1) continue;
                    ll dP = d1+d2-baseP;
                    if(P+dP > PMAX) continue;
                    int dWv = (grow>0)? (Fp(t)-FX) : (FX-Fp(t-1));
                    double obj = (double)dWv - lamLen*(double)dP;
                    if(obj > bestObj+1e-9){ bestObj=obj; bestT=t; }
                }
                if(bestT!=X){
                    ll d1=llabs(bestT-ca), d2=llabs(bestT-cb);
                    P += d1+d2-baseP;
                    if(vert){ VX[i]=bestT; VX[i1]=bestT; } else { VY[i]=bestT; VY[i1]=bestT; }
                    any=true;
                }
            }
            passes++;
            if(!any){
                if(P < (ll)(PMAX*pFull) && lamLen > 1e-7 && stall<12){ lamLen *= lamDecay; stall++; continue; }
                break;
            }
        }
        if(false) fprintf(stderr,"polish passes=%d P=%lld t=%.3f\n", passes, P, nowsec());
        if(!validPoly(VX,VY)){ VX=BX; VY=BY; }
    }
    if(!validPoly(VX,VY)){
        VX = {0,0,100000,100000}; VY = {0,100000,100000,0};
        if(!validPoly(VX,VY)){ VX={0,1,1,0}; VY={0,0,1,1}; }
    }
    {
        string s2; s2.reserve(VX.size()*16);
        s2 += to_string(VX.size()); s2 += "\n";
        for(size_t i=0;i<VX.size();i++){ s2 += to_string(VX[i]); s2 += " "; s2 += to_string(VY[i]); s2 += "\n"; }
        fwrite(s2.data(),1,s2.size(),stdout);
    }
    return 0;
}
# EVOLVE-BLOCK-END
