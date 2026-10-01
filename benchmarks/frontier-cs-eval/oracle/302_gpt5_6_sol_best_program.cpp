// EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll INFLL = (1LL << 62);

struct FastRng {
    using result_type=uint64_t;
    uint64_t x;
    explicit FastRng(uint64_t seed):x(seed){}
    static constexpr result_type min(){return 0;}
    static constexpr result_type max(){return UINT64_MAX;}
    result_type operator()(){x^=x<<7;x^=x>>9;return x;}
};

struct AC {
    struct Node {
        array<int, 26> nx;
        int link = 0;
        ll out = 0, risk = 0;
        Node() { nx.fill(-1); }
    };
    int k;
    vector<Node> t;
    AC(int kk=0): k(kk), t(1) {}
    void add(const string &s, ll w) {
        int v=0;
        for(int at=0;at<(int)s.size();at++) {
            char ch=s[at];
            int c=ch-'a';
            if(t[v].nx[c]<0) { t[v].nx[c]=t.size(); t.emplace_back(); }
            v=t[v].nx[c];
            int left=(int)s.size()-1-at;
            if(left>0) t[v].risk += max(1LL,w>>min(left,20));
        }
        t[v].out += w;
    }
    void build() {
        queue<int> q;
        for(int c=0;c<k;c++) {
            int u=t[0].nx[c];
            if(u<0) t[0].nx[c]=0;
            else q.push(u);
        }
        while(!q.empty()) {
            int v=q.front(); q.pop();
            t[v].out += t[t[v].link].out;
            t[v].risk += t[t[v].link].risk;
            for(int c=0;c<k;c++) {
                int u=t[v].nx[c];
                if(u<0) t[v].nx[c]=t[t[v].link].nx[c];
                else { t[u].link=t[t[v].link].nx[c]; q.push(u); }
            }
        }
    }
};

// A reversed trie makes the exact contribution of motifs ending at a position
// available after inspecting at most ten characters.
struct RevTrie {
    struct Node {
        array<int,26> nx;
        ll term=0;
        Node(){ nx.fill(-1); }
    };
    vector<Node> t{1};
    void add(const string &s,ll w) {
        int v=0;
        for(auto it=s.rbegin();it!=s.rend();++it) {
            int c=*it-'a';
            if(t[v].nx[c]<0){ t[v].nx[c]=t.size(); t.emplace_back(); }
            v=t[v].nx[c];
        }
        t[v].term+=w;
    }
    ll ending(const string &s,int p) const {
        int v=0;
        ll z=0;
        for(int q=p, steps=0;q>=0 && steps<10;q--,steps++) {
            int c=s[q]-'a';
            v=t[v].nx[c];
            if(v<0) break;
            z+=t[v].term;
        }
        return z;
    }
};

struct MCMF {
    struct E { int to, rev; ll cap, cost, original; };
    int N;
    vector<vector<E>> g;
    MCMF(int n):N(n),g(n){}
    int edge(int v,int u,ll cap,ll cost) {
        int id=g[v].size();
        E a{u,(int)g[u].size(),cap,cost,cap};
        E b{v,id,0,-cost,0};
        g[v].push_back(a); g[u].push_back(b);
        return id;
    }
    pair<ll,ll> run(int S,int T,ll need) {
        vector<ll> pot(N),d(N);
        vector<int> pv(N),pe(N);
        ll flow=0,cost=0;
        while(flow<need) {
            fill(d.begin(),d.end(),INFLL); d[S]=0;
            priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<pair<ll,int>>> pq;
            pq.push({0,S});
            while(!pq.empty()) {
                auto [dd,v]=pq.top();pq.pop(); if(dd!=d[v])continue;
                for(int i=0;i<(int)g[v].size();i++) {
                    E const &e=g[v][i]; if(!e.cap)continue;
                    ll nd=dd+e.cost+pot[v]-pot[e.to];
                    if(nd<d[e.to]){d[e.to]=nd;pv[e.to]=v;pe[e.to]=i;pq.push({nd,e.to});}
                }
            }
            if(d[T]==INFLL) break;
            for(int v=0;v<N;v++) if(d[v]<INFLL) pot[v]+=d[v];
            ll add=need-flow;
            for(int v=T;v!=S;v=pv[v]) add=min(add,g[pv[v]][pe[v]].cap);
            for(int v=T;v!=S;v=pv[v]) {
                E &e=g[pv[v]][pe[v]];
                cost += add*e.cost; e.cap-=add; g[v][e.rev].cap+=add;
            }
            flow+=add;
        }
        return {flow,cost};
    }
};

struct FlowSol {
    ll cost=INFLL;
    int st=0,en=0;
    vector<vector<ll>> x;
};

static FlowSol transportation(const vector<int>& cnt,const vector<vector<ll>>& C,int st,int en) {
    int k=cnt.size(), S=2*k, T=S+1;
    MCMF mf(T+1);
    vector<vector<int>> id(k,vector<int>(k));
    ll need=0;
    for(int i=0;i<k;i++) {
        ll z=cnt[i]-(i==en); need+=z; mf.edge(S,i,z,0);
    }
    for(int i=0;i<k;i++) for(int j=0;j<k;j++) id[i][j]=mf.edge(i,k+j,(ll)4e18,C[i][j]);
    for(int j=0;j<k;j++) mf.edge(k+j,T,cnt[j]-(j==st),0);
    auto [f,cost]=mf.run(S,T,need);
    FlowSol r; r.st=st;r.en=en;r.cost=(f==need?cost:INFLL);r.x.assign(k,vector<ll>(k));
    if(f==need) for(int i=0;i<k;i++) for(int j=0;j<k;j++) {
        auto const &e=mf.g[i][id[i][j]]; r.x[i][j]=e.original-e.cap;
    }
    return r;
}

struct DSU {
    vector<int> p;
    DSU(int n):p(n){iota(p.begin(),p.end(),0);}
    int f(int x){return p[x]==x?x:p[x]=f(p[x]);}
    void un(int a,int b){a=f(a);b=f(b);if(a!=b)p[a]=b;}
};

static bool connectFlow(FlowSol &f,const vector<vector<ll>>& C) {
    int k=f.x.size();
    for(int rounds=0;rounds<k;rounds++) {
        DSU d(k);
        for(int i=0;i<k;i++)for(int j=0;j<k;j++)if(f.x[i][j])d.un(i,j);
        int root=d.f(0); bool done=true;
        for(int i=1;i<k;i++)if(d.f(i)!=root){done=false;break;}
        if(done) return true;
        vector<pair<int,int>> edges;
        for(int a=0;a<k;a++)for(int b=0;b<k;b++)if(f.x[a][b])edges.push_back({a,b});
        ll bd=INFLL; int ba=-1,bb=-1,bc=-1,be=-1;
        for(auto [a,b]:edges) for(auto [c,e]:edges) if(d.f(a)!=d.f(c)) {
                ll z=C[a][e]+C[c][b]-C[a][b]-C[c][e];
                if(z<bd){bd=z;ba=a;bb=b;bc=c;be=e;}
        }
        if(ba<0) return false;
        f.x[ba][bb]--;f.x[bc][be]--;f.x[ba][be]++;f.x[bc][bb]++;
        f.cost+=bd;
    }
    return false;
}

static string eulerGreedy(const FlowSol &fs,const AC &ac,int mode,FastRng &rng,int n) {
    int k=fs.x.size(), root=fs.en;
    vector<vector<ll>> rem=fs.x;
    vector<int> par(k,-1),seen(k); seen[root]=1;
    queue<int> q;q.push(root);
    while(!q.empty()) {
        int u=q.front();q.pop();
        vector<int> vv;
        for(int v=0;v<k;v++)if(!seen[v] && rem[v][u])vv.push_back(v);
        if(mode==2) shuffle(vv.begin(),vv.end(),rng);
        for(int v:vv){seen[v]=1;par[v]=u;q.push(v);}
    }
    for(int v=0;v<k;v++)if(v!=root) {
        if(par[v]<0 || rem[v][par[v]]==0) return {};
        rem[v][par[v]]--;
    }
    vector<char> reserve(k,1);reserve[root]=0;
    vector<ll> rows(k);
    for(int i=0;i<k;i++)for(int j=0;j<k;j++)rows[i]+=rem[i][j];
    string s;s.reserve(n);
    int cur=fs.st,state=ac.t[0].nx[cur];s.push_back(char('a'+cur));
    for(int pos=1;pos<n;pos++) {
        int go=-1;
        if(rows[cur]) {
            if(mode==2 && (rng()&3)==0) {
                ll take=rng()%rows[cur];
                for(int j=0;j<k;j++)if(rem[cur][j]){if(take<(ll)rem[cur][j]){go=j;break;}take-=rem[cur][j];}
            } else {
                ll best=INFLL;
                for(int j=0;j<k;j++)if(rem[cur][j]) {
                    int ns=ac.t[state].nx[j];
                    ll val=ac.t[ns].out;
                    if(mode==1) {
                        ll la=INFLL;
                        for(int z=0;z<k;z++) if(rem[j][z] || (reserve[j]&&par[j]==z))
                            la=min(la,ac.t[ac.t[ns].nx[z]].out);
                        if(la<INFLL) val += la/2;
                    }
                    if(mode==3) val += ac.t[ns].risk/4;
                    if(mode==4) val += ac.t[ns].risk;
                    if(val<best || (val==best && (rng()&1))){best=val;go=j;}
                }
            }
            if(go<0)return {};
            rem[cur][go]--;rows[cur]--;
        } else {
            if(!reserve[cur]) return {};
            go=par[cur];reserve[cur]=0;
        }
        cur=go;state=ac.t[state].nx[cur];s.push_back(char('a'+cur));
    }
    if(cur!=fs.en)return {};
    return s;
}

static ll fullScore(const string &s,const vector<vector<ll>>& W,const AC& ac) {
    ll z=0;int state=0;
    for(int i=0;i<(int)s.size();i++) {
        int c=s[i]-'a';
        if(i)z+=W[s[i-1]-'a'][c];
        state=ac.t[state].nx[c];z+=ac.t[state].out;
    }
    return z;
}

static void consider(string s,string &best,ll &bestScore,const vector<vector<ll>>&W,const AC&ac) {
    if(s.empty())return;
    ll z=fullScore(s,W,ac);
    if(z<bestScore){bestScore=z;best=move(s);}
}

// Exact multiset/automaton DP when the complete state space is genuinely small.
// The mixed-radix count code is topologically ordered because every transition
// increases it by a positive radix multiplier.
static string exactSmall(const vector<int>&cnt,const vector<vector<ll>>&W,const AC&ac) {
    int k=cnt.size(), A=ac.t.size(), B=A*(k+1);
    uint64_t prod=1;
    vector<uint64_t> mul(k);
    for(int c=0;c<k;c++) {
        mul[c]=prod;
        if(prod>3000000ULL/(uint64_t)(cnt[c]+1))return {};
        prod*=cnt[c]+1;
    }
    if(prod>3000000ULL/(uint64_t)B)return {};
    uint64_t cells=prod*B;
    if(cells>3000000ULL)return {};
    vector<ll> dp(cells,INFLL);
    vector<uint32_t> par(cells,UINT32_MAX);
    auto ix=[&](uint64_t code,int state,int last){return code*(uint64_t)B+state*(k+1)+last;};
    dp[ix(0,0,k)]=0;
    vector<int> used(k);
    for(uint64_t code=0;code<prod;code++) {
        for(int c=0;c<k;c++)used[c]=(code/mul[c])%(cnt[c]+1);
        uint64_t base=code*B;
        for(int st=0;st<A;st++)for(int last=0;last<=k;last++) {
            uint64_t at=base+st*(k+1)+last;
            ll old=dp[at];if(old==INFLL)continue;
            for(int c=0;c<k;c++)if(used[c]<cnt[c]) {
                int ns=ac.t[st].nx[c];
                uint64_t ni=ix(code+mul[c],ns,c);
                ll z=old+ac.t[ns].out+(last<k?W[last][c]:0);
                if(z<dp[ni]){dp[ni]=z;par[ni]=(uint32_t)at;}
            }
        }
    }
    uint64_t code=prod-1,best=UINT64_MAX;ll bv=INFLL;
    for(int st=0;st<A;st++)for(int last=0;last<k;last++) {
        uint64_t at=ix(code,st,last);if(dp[at]<bv){bv=dp[at];best=at;}
    }
    if(best==UINT64_MAX)return {};
    string s;s.reserve(accumulate(cnt.begin(),cnt.end(),0));
    while(code) {
        int stateLast=best%B, ch=stateLast%(k+1);
        if(ch>=k)return {};
        s.push_back(char('a'+ch));code-=mul[ch];best=par[best];
    }
    reverse(s.begin(),s.end());return s;
}

static string exactBlockPath(const vector<int>&cnt,const vector<vector<ll>>&C) {
    int k=cnt.size();if(k>18)return {};
    int M=1<<k;
    vector<ll> dp((size_t)M*k,INFLL);
    vector<int8_t> par((size_t)M*k,-1);
    for(int i=0;i<k;i++)dp[(1<<i)*k+i]=0;
    for(int mask=1;mask<M;mask++)for(int last=0;last<k;last++) {
        ll old=dp[(size_t)mask*k+last];if(old==INFLL)continue;
        int left=(M-1)^mask;
        while(left){int b=__builtin_ctz(left);left&=left-1;ll z=old+C[last][b];
            size_t at=(size_t)(mask|(1<<b))*k+b;
            if(z<dp[at]){dp[at]=z;par[at]=last;}
        }
    }
    int mask=M-1,last=min_element(dp.begin()+(size_t)mask*k,dp.begin()+(size_t)(mask+1)*k)-
                              (dp.begin()+(size_t)mask*k);
    vector<int> ord;
    while(mask){ord.push_back(last);int p=par[(size_t)mask*k+last];mask^=1<<last;last=p;}
    reverse(ord.begin(),ord.end());string s;
    s.reserve(accumulate(cnt.begin(),cnt.end(),0));for(int c:ord)s.append(cnt[c],char('a'+c));return s;
}

static ll intervalCost(const string&s,int l,int r,const vector<vector<ll>>&W,const AC&ac) {
    if(l>r)return 0;
    int state=0;
    for(int p=max(0,l-9);p<l;p++)state=ac.t[state].nx[s[p]-'a'];
    ll z=0;
    for(int p=l;p<=r;p++) {
        int c=s[p]-'a';if(p)z+=W[s[p-1]-'a'][c];
        state=ac.t[state].nx[c];z+=ac.t[state].out;
    }
    return z;
}

static void localSearch(string &s,ll &score,const vector<vector<ll>>&W,const AC&ac,const RevTrie&rt,
                        FastRng&rng,chrono::steady_clock::time_point deadline) {
    int n=s.size(); if(n<2)return;
    ll bestZ=score;
    struct Move{int i,j,len;};
    vector<Move> undo;undo.reserve(100000);
    ll avg=max(1LL,score/max(1,n));
    vector<ll> impact(n);
    vector<unsigned char> valid(n);
    array<uint64_t,256> metro;
    for(int q=0;q<256;q++)metro[q]=(uint64_t)(expl(-(long double)q/16.0L)*(long double)UINT64_MAX);
    auto uphill=[&](ll delta,double temp){
        int q=(int)min(255.0,(double)delta*16.0/temp);
        return rng()<metro[q];
    };
    auto getImpact=[&](int p){
        if(!valid[p]){impact[p]=rt.ending(s,p)+(p?W[s[p-1]-'a'][s[p]-'a']:0);valid[p]=1;}
        return impact[p];
    };
    auto invalidate=[&](int a,int b){for(int p=a;p<=b&&p<n;p++)valid[p]=0;};
    int it=0;uint64_t attempts=0;
    while(true) {
        if((attempts++&255)==0 && chrono::steady_clock::now()>=deadline)break;
        int i,j;
        uint64_t typ=rng()%10;
        if(typ>=5 && n>=6) {
            int len=2+rng()%min(9,n/3);
            i=rng()%(n-len+1);j=rng()%(n-len+1);
            if(i>j)swap(i,j);
            if(i+len>j)continue;
            int ei=min(n-1,i+len+8),ej=min(n-1,j+len+8);
            auto rangeCost=[&](int a,int b){return intervalCost(s,a,b,W,ac);};
            auto part=[&](){return rangeCost(i,ei)+(j>ei?rangeCost(j,ej):rangeCost(ei+1,ej));};
            ll old=part();for(int q=0;q<len;q++)swap(s[i+q],s[j+q]);ll delta=part()-old;
            double progress=min(1.0,it/250000.0);
            double temp=(double)avg*(0.02*(1.0-progress)+0.0001);
            bool accept=delta<=0 || uphill(delta,temp);
            if(accept){score+=delta;undo.push_back({i,j,len});invalidate(i,ei);invalidate(j,ej);if(score<bestZ){bestZ=score;undo.clear();}}
            else for(int q=0;q<len;q++)swap(s[i+q],s[j+q]);
            it++;continue;
        }
        if(typ<1) { i=rng()%n; int rad=min(12,n-1); j=max(0,min(n-1,i+(int)(rng()%(2*rad+1))-rad)); }
        else if(typ<5) {
            i=rng()%n;
            for(int q=0;q<3;q++) {
                int x=rng()%n;
                ll ci=getImpact(i),cx=getImpact(x);
                if(cx>ci)i=x;
            }
            j=rng()%n;
            for(int q=0;q<4;q++){int x=rng()%n;if(getImpact(x)>getImpact(j))j=x;}
        } else {i=rng()%n;j=rng()%n;}
        if(i==j || s[i]==s[j])continue;
        if(i>j)swap(i,j);
        int ei=min(n-1,i+9),ej=min(n-1,j+9);
        auto rangeCost=[&](int a,int b){return intervalCost(s,a,b,W,ac);};
        auto part=[&](){return rangeCost(i,ei)+(j>ei?rangeCost(j,ej):rangeCost(ei+1,ej));};
        ll old=part();swap(s[i],s[j]);ll nw=part(),delta=nw-old;
        double progress=min(1.0,it/250000.0);
        double temp=(double)avg*(0.025*(1.0-progress)+0.0002);
        bool accept=delta<=0 || uphill(delta,temp);
        if(accept){score+=delta;undo.push_back({i,j,0});invalidate(i,ei);invalidate(j,ej);if(score<bestZ){bestZ=score;undo.clear();}}
        else swap(s[i],s[j]);
        it++;
    }
    for(auto it=undo.rbegin();it!=undo.rend();++it) {
        if(it->len)for(int q=0;q<it->len;q++)swap(s[it->i+q],s[it->j+q]);
        else swap(s[it->i],s[it->j]);
    }
    score=bestZ;
}

int main(){
    ios::sync_with_stdio(false);cin.tie(nullptr);
    auto started=chrono::steady_clock::now();
    int n,k;if(!(cin>>n>>k))return 0;
    vector<int> cnt(k);for(int&i:cnt)cin>>i;
    vector<vector<ll>> W(k,vector<ll>(k)),C,Csoft;
    for(auto&r:W)for(ll&x:r)cin>>x;
    C=Csoft=W;
    int m;cin>>m;
    AC ac(k);RevTrie rt;
    for(int z=0;z<m;z++){
        string p;ll w;cin>>p>>w;ac.add(p,w);rt.add(p,w);
        if(p.size()==2) {
            C[p[0]-'a'][p[1]-'a']+=w;
            Csoft[p[0]-'a'][p[1]-'a']+=w;
        } else {
            ll share=max(1LL,w/(ll)(p.size()-1));
            for(int i=1;i<(int)p.size();i++)Csoft[p[i-1]-'a'][p[i]-'a']+=share;
        }
    }
    ac.build();
    FastRng rng(0x9e3779b97f4a7c15ULL+n*1009ULL+k);
    string best;ll bestScore=INFLL;
    string sorted;
    for(int i=0;i<k;i++)sorted.append(cnt[i],char('a'+i));
    consider(sorted,best,bestScore,W,ac);
    string rev;
    for(int i=k-1;i>=0;i--)rev.append(cnt[i],char('a'+i));
    consider(rev,best,bestScore,W,ac);
    consider(exactSmall(cnt,W,ac),best,bestScore,W,ac);
    consider(exactBlockPath(cnt,C),best,bestScore,W,ac);

    auto tryModel = [&](const vector<vector<ll>>& CM,int keep,int use,int stride) {
        vector<FlowSol> pool;
        for(int st=0;st<k;st++)for(int en=0;en<k;en++) {
            if((st*k+en)%stride)continue;
            if(k>1 && st==en && cnt[st]==1)continue;
            FlowSol f=transportation(cnt,CM,st,en);
            if(f.cost!=INFLL)pool.push_back(move(f));
        }
        sort(pool.begin(),pool.end(),[](auto const&a,auto const&b){return a.cost<b.cost;});
        if((int)pool.size()>keep)pool.resize(keep);
        vector<FlowSol> usable;
        for(auto &f:pool)if(connectFlow(f,CM))usable.push_back(move(f));
        sort(usable.begin(),usable.end(),[](auto const&a,auto const&b){return a.cost<b.cost;});
        if((int)usable.size()>use)usable.resize(use);
        for(auto &f:usable)for(int mode=0;mode<5;mode++)
            consider(eulerGreedy(f,ac,mode,rng,n),best,bestScore,W,ac);
    };
    tryModel(C,24,8,1);
    if(Csoft!=C)tryModel(Csoft,16,5,8);

    // A few block permutations are useful when long repeated motifs dominate and
    // splitting blocks would be counterproductive.
    for(int first=0;first<k;first++) {
        vector<char> used(k);vector<int> ord{first};used[first]=1;
        while((int)ord.size()<k){int v=ord.back(),bj=-1;ll bv=INFLL;for(int j=0;j<k;j++)if(!used[j]&&C[v][j]<bv){bv=C[v][j];bj=j;}used[bj]=1;ord.push_back(bj);}
        string z;z.reserve(n);for(int v:ord)z.append(cnt[v],char('a'+v));consider(move(z),best,bestScore,W,ac);
    }

    auto deadline=started+chrono::milliseconds(1970);
    localSearch(best,bestScore,W,ac,rt,rng,deadline);
    cout<<best<<'\n';
}
// EVOLVE-BLOCK-END
