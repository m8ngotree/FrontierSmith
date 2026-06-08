#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef double ld;
typedef vector<ld> VD;
typedef vector<VD> VVD;
const ld EPS = 1e-9;

// KACTL-style LP: maximize c^T x s.t. Ax <= b, x >= 0
struct LPSolver {
    int m, n;
    vector<int> N, B;
    VVD D;
    LPSolver(VVD& A, VD& b, VD& c)
        : m(A.size()), n(c.size()), N(n+1), B(m), D(m+2, VD(n+2,0)) {
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) D[i][j]=A[i][j];
        for(int i=0;i<m;i++){B[i]=n+i;D[i][n]=-1;D[i][n+1]=b[i];}
        for(int j=0;j<n;j++){N[j]=j;D[m][j]=-c[j];}
        N[n]=-1; D[m+1][n]=1;
    }
    void pivot(int r, int s){
        ld inv=1.0/D[r][s];
        for(int i=0;i<=m+1;i++) if(i!=r){
            for(int j=0;j<=n+1;j++) if(j!=s) D[i][j]-=D[i][s]*D[r][j]*inv;
            D[i][s]*=-inv;
        }
        for(int j=0;j<=n+1;j++) if(j!=s) D[r][j]*=inv;
        D[r][s]=inv;
        swap(B[r],N[s]);
    }
    bool simplex(int phase){
        int x=phase==1?m+1:m;
        for(;;){
            int s=-1;
            for(int j=0;j<=n;j++){
                if(phase==2&&N[j]==-1) continue;
                if(s==-1||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&N[j]<N[s])) s=j;
            }
            if(D[x][s]>=-EPS) return true;
            int r=-1;
            for(int i=0;i<m;i++){
                if(D[i][s]<=EPS) continue;
                if(r==-1||D[i][n+1]/D[i][s]<D[r][n+1]/D[r][s]||
                   (D[i][n+1]/D[i][s]==D[r][n+1]/D[r][s]&&B[i]<B[r])) r=i;
            }
            if(r==-1) return false;
            pivot(r,s);
        }
    }
    ld solve(){
        int r=0;
        for(int i=1;i<m;i++) if(D[i][n+1]<D[r][n+1]) r=i;
        if(D[r][n+1]<-EPS){
            pivot(r,n);
            if(!simplex(1)||D[m+1][n+1]<-EPS) return -1e18;
            for(int i=0;i<m;i++) if(B[i]==-1){
                int s2=-1;
                for(int j=0;j<=n;j++)
                    if(s2==-1||D[i][j]<D[i][s2]||(D[i][j]==D[i][s2]&&N[j]<N[s2])) s2=j;
                pivot(i,s2);
            }
        }
        if(!simplex(2)) return 1e18;
        return D[m][n+1];
    }
};

int D, N, K;
vector<VD> X;
vector<long long> P;

bool isDominated(const vector<int>& S, int k){
    if(S.empty()) return false;
    int m=(int)S.size();
    // Quick: any single point dominates k?
    for(int s:S){
        bool ok=true;
        for(int c=0;c<D;c++) if(X[s][c]<X[k][c]){ok=false;break;}
        if(ok) return true;
    }
    if(m==1) return false;
    // Quick: for each coord, is there any point in S with x[s][c] >= x[k][c]?
    for(int c=0;c<D;c++){
        bool any=false;
        for(int s:S) if(X[s][c]>=X[k][c]){any=true;break;}
        if(!any) return false;
    }
    // LP: maximize t' (= t + SH, shift so t' >= 0) s.t.
    // -sum lambda_i*(x_i[c]-x_k[c]) + t' <= SH  for each c
    // sum lambda <= 1
    // -sum lambda <= -1
    // t' <= 2*SH
    // lambda_i >= 0, t' >= 0
    ld SH = 2e9;
    int nv = m+1;
    VVD A; VD b;
    for(int c=0;c<D;c++){
        VD row(nv,0);
        for(int i=0;i<m;i++) row[i]=-(X[S[i]][c]-X[k][c]);
        row[m]=1;
        A.push_back(row); b.push_back(SH);
    }
    {VD row(nv,0);for(int i=0;i<m;i++)row[i]=1;A.push_back(row);b.push_back(1);}
    {VD row(nv,0);for(int i=0;i<m;i++)row[i]=-1;A.push_back(row);b.push_back(-1);}
    {VD row(nv,0);row[m]=1;A.push_back(row);b.push_back(2*SH);}
    VD c(nv,0); c[m]=1;
    LPSolver lp(A,b,c);
    ld val=lp.solve();
    return val >= SH - 1e-6;
}

long long computeClosure(const vector<int>& seeds){
    vector<bool> activated(N,false);
    for(int s:seeds) activated[s]=true;
    bool changed=true;
    while(changed){
        changed=false;
        vector<int> cur;
        for(int i=0;i<N;i++) if(activated[i]) cur.push_back(i);
        for(int k=0;k<N;k++){
            if(activated[k]) continue;
            if(isDominated(cur,k)){
                activated[k]=true;
                changed=true;
            }
        }
    }
    long long total=0;
    for(int i=0;i<N;i++) if(activated[i]) total+=P[i];
    return total;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    // Read input
    D = inf.readInt();
    N = inf.readInt();
    K = inf.readInt();
    X.resize(N, VD(D));
    P.resize(N);
    for(int i=0;i<N;i++){
        for(int c=0;c<D;c++) X[i][c]=(ld)inf.readInt();
        P[i]=(long long)inf.readInt();
    }

    // Read participant output
    int m = ouf.readInt();
    if(m < 0 || m > K)
        quitf(_wa, "m=%d out of range [0,%d]", m, K);

    vector<int> seeds;
    set<int> seedSet;
    for(int i=0;i<m;i++){
        int idx = ouf.readInt();
        if(idx < 1 || idx > N)
            quitf(_wa, "index %d out of range [1,%d]", idx, N);
        if(seedSet.count(idx))
            quitf(_wa, "duplicate index %d", idx);
        seedSet.insert(idx);
        seeds.push_back(idx-1);
    }
    if(!ouf.seekEof())
        quitf(_wa, "extra output after seeds");

    // Compute F(S) for participant
    long long R = computeClosure(seeds);
    long long Rplus = max(0LL, R);

    // Compute U = sum max(P[i], 0)
    long long U = 0;
    for(int i=0;i<N;i++) U += max(0LL, P[i]);

    // Compute B = max(0, max_j F({j}))
    long long B = 0;
    for(int j=0;j<N;j++){
        vector<int> single = {j};
        long long sc = computeClosure(single);
        if(sc > B) B = sc;
    }
    if(B < 0) B = 0;

    // Normalize
    double ratio;
    if(U == B){
        ratio = (Rplus == B) ? 1.0 : 0.0;
    } else {
        ratio = (double)(Rplus - B) / (double)(U - B);
    }
    if(ratio < 0.0) ratio = 0.0;
    if(ratio > 1.0) ratio = 1.0;

    quitp(ratio, "OK R=%lld B=%lld U=%lld Ratio: %.6f", R, B, U, ratio);
    return 0;
}