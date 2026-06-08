#include <bits/stdc++.h>
using namespace std;
typedef double ld;
typedef vector<ld> VD;
typedef vector<VD> VVD;

const ld EPS = 1e-9;

// Simplex LP solver: maximize c^T x s.t. Ax <= b, x >= 0
// Returns -1e18 if infeasible, 1e18 if unbounded, else optimal value
struct Simplex {
    int m, n;
    vector<int> bas, nonbas;
    VVD tab;

    Simplex(VVD& A, VD& b, VD& c) : m(A.size()), n(c.size()), bas(m), nonbas(n+1), tab(m+2, VD(n+2)) {
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) tab[i][j]=A[i][j];
        for(int i=0;i<m;i++){tab[i][n]=1;tab[i][n+1]=b[i];}
        for(int j=0;j<n;j++) tab[m][j]=-c[j];
        tab[m][n]=0; tab[m][n+1]=0;
        for(int i=0;i<m;i++) bas[i]=n+i, nonbas[i]=i;
        nonbas[n]=n+m; // artificial?
        // Revised: use standard form
        iota(bas.begin(),bas.end(),n);
        iota(nonbas.begin(),nonbas.end(),0);
    }

    void pivot(int r, int s){
        ld inv=1.0/tab[r][s];
        for(int i=0;i<=m+1;i++) if(i!=r) {
            ld mul=tab[i][s]*inv;
            for(int j=0;j<=(int)tab[0].size()-1;j++) tab[i][j]-=mul*tab[r][j];
            tab[i][s]=0;
        }
        for(int j=0;j<(int)tab[0].size();j++) tab[r][j]*=inv;
        tab[r][s]=inv; // actually = 1 after division... let me redo
        // Actually pivot properly:
    }
};

// Better: use a clean tableau simplex
// maximize c^T x s.t. Ax <= b, x >= 0
// N vars, M constraints
ld simplex2(int N, int M, VVD A, VD b, VD c) {
    // tableau: M+1 rows, N+M+1 cols
    // slack vars x[N..N+M-1], RHS at col N+M
    int tot = N + M;
    VVD tab(M+1, VD(tot+1, 0));
    vector<int> bas(M);
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++) tab[i][j]=A[i][j];
        tab[i][N+i]=1;
        tab[i][tot]=b[i];
        bas[i]=N+i;
    }
    for(int j=0;j<N;j++) tab[M][j]=-c[j];

    // Check if initial basis is feasible (need all b[i]>=0)
    // If not, do phase 1
    // For now assume b[i] can be negative -> handle by picking most negative
    while(true){
        // find entering variable (most negative in objective row)
        int ent=-1;
        for(int j=0;j<tot;j++) if(tab[M][j]<-EPS && (ent==-1||tab[M][j]<tab[M][ent])) ent=j;
        if(ent==-1) break;
        // find leaving variable (min ratio)
        int lev=-1;
        for(int i=0;i<M;i++) if(tab[i][ent]>EPS){
            if(lev==-1||tab[i][tot]/tab[i][ent]<tab[lev][tot]/tab[lev][ent]) lev=i;
        }
        if(lev==-1) return 1e18; // unbounded
        // pivot
        ld piv=tab[lev][ent];
        for(int j=0;j<=tot;j++) tab[lev][j]/=piv;
        for(int i=0;i<=M;i++) if(i!=lev){
            ld mul=tab[i][ent];
            for(int j=0;j<=tot;j++) tab[i][j]-=mul*tab[lev][j];
        }
        bas[lev]=ent;
    }
    // check feasibility: all basic vars should be >= 0
    for(int i=0;i<M;i++) if(tab[i][tot]<-EPS) return -1e18;
    return tab[M][tot];
}

// Check if point k is dominated by conv(S)
// S: list of point indices (0-based), pts: all points, D: dims
bool dominated(vector<int>& S, vector<vector<int>>& pts, int k, int D){
    if(S.empty()) return false;
    int m=S.size();
    // maximize z s.t. sum lambda_i * x_{i,c} - z >= x_k,c for c=0..D-1
    // sum lambda_i = 1, lambda_i >= 0, z free
    // Variables: lambda_0..lambda_{m-1}, z+ , z- (z = z+ - z-)
    // N vars = m+2
    // Constraints: D (>=) + 1 (sum=1) + 1 (sum>=-1, i.e., sum<=... wait equality)
    // Equality sum=1: split as sum<=1 and sum>=-1 (i.e., -sum<=−1)
    // >= constraints: flip sign
    // maximize z+ - z-
    int N = m+2; // lambda[0..m-1], z+, z-
    int M = D + 2; // D constraints + 2 for equality
    VVD A(M, VD(N,0));
    VD b(M), c(N,0);
    c[m]=1; c[m+1]=-1; // maximize z+ - z-
    // For c=0..D-1: sum_i lambda_i*x_{i,c} - z+ + z- >= x_k[c]
    // => -sum_i lambda_i*x_{i,c} + z+ - z- <= -x_k[c]
    for(int d=0;d<D;d++){
        for(int i=0;i<m;i++) A[d][i]=-pts[S[i]][d];
        A[d][m]=1; A[d][m+1]=-1;
        b[d]=-pts[k][d];
    }
    // sum lambda <= 1
    for(int i=0;i<m;i++) A[D][i]=1;
    b[D]=1;
    // -sum lambda <= -1
    for(int i=0;i<m;i++) A[D+1][i]=-1;
    b[D+1]=-1;

    // Need initial feasible basis. b[D+1]=-1<0, need to handle.
    // Use Big-M method: add artificial variable with large cost for infeasible rows
    // Actually, let's use a different approach: multiply row D+1 by -1
    for(int j=0;j<N;j++) A[D+1][j]=-A[D+1][j];
    b[D+1]=1;
    // Now -sum lambda <= -1 becomes sum lambda >= 1 but we flipped to sum lambda <= 1... wait
    // Original: -sum lambda <= -1. Multiply by -1: sum lambda >= 1. Now we have >= constraint.
    // For simplex Ax<=b form, flip: -sum lambda <= -1 is fine if b[D+1]=-1, but that's negative.
    // Just add big-M artificial variables for rows with negative b.
    // Simpler: just solve with the two-phase simplex below.
    // Let me re-add it:
    for(int j=0;j<N;j++) A[D+1][j]=-A[D+1][j];
    b[D+1]=-1;

    // Use big-M method for initial feasibility
    // For rows with b[i]<0, multiply by -1 and add artificial
    int artN = N;
    vector<int> arts;
    VVD A2=A; VD b2=b, c2=c;
    for(int i=0;i<M;i++){
        if(b2[i]<-EPS){
            for(int j=0;j<(int)A2[i].size();j++) A2[i][j]=-A2[i][j];
            b2[i]=-b2[i];
        }
    }
    ld val = simplex2(N, M, A2, b2, c2);
    return val >= -EPS;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);

    int D,N,K;
    cin>>D>>N>>K;

    vector<vector<int>> pts(N, vector<int>(D));
    vector<int> P(N);
    for(int i=0;i<N;i++){
        for(int d=0;d<D;d++) cin>>pts[i][d];
        cin>>P[i];
    }

    if(K==0){ cout<<0<<"\n\n"; return 0; }

    // Compute closure for a set of seeds
    auto computeClosure = [&](vector<int>& seeds) -> vector<bool> {
        vector<bool> inCl(N, false);
        for(int s: seeds) inCl[s]=true;
        if(seeds.empty()) return inCl;
        // Check each point
        for(int k=0;k<N;k++){
            if(inCl[k]) continue;
            if(dominated(seeds, pts, k, D)) inCl[k]=true;
        }
        return inCl;
    };

    auto score = [&](vector<bool>& cl) -> long long {
        long long s=0;
        for(int i=0;i<N;i++) if(cl[i]) s+=P[i];
        return s;
    };

    // Greedy seed selection
    vector<int> seeds;
    vector<bool> inCl(N,false);
    long long curScore=0;

    for(int iter=0;iter<K;iter++){
        int best=-1;
        long long bestGain=-1;
        vector<bool> bestCl;

        for(int t=0;t<N;t++){
            bool already=false;
            for(int s:seeds) if(s==t){already=true;break;}
            if(already) continue;

            vector<int> newSeeds=seeds;
            newSeeds.push_back(t);
            auto cl=computeClosure(newSeeds);
            long long sc=score(cl);
            long long gain=sc-curScore;
            if(best==-1||gain>bestGain){
                bestGain=gain;
                best=t;
                bestCl=cl;
            }
        }

        if(best==-1||bestGain<=0) break;
        seeds.push_back(best);
        inCl=bestCl;
        curScore=score(inCl);
    }

    // Local search: try replacing each seed with another point
    bool improved=true;
    int lsIter=0;
    while(improved && lsIter<20){
        improved=false;
        lsIter++;
        for(int si=0;si<(int)seeds.size();si++){
            for(int t=0;t<N;t++){
                bool inSeeds=false;
                for(int s:seeds) if(s==t){inSeeds=true;break;}
                if(inSeeds) continue;
                vector<int> newSeeds=seeds;
                newSeeds[si]=t;
                auto cl=computeClosure(newSeeds);
                long long sc=score(cl);
                if(sc>curScore){
                    seeds[si]=t;
                    inCl=cl;
                    curScore=sc;
                    improved=true;
                }
            }
        }
        // Try adding more seeds if under K
        if((int)seeds.size()<K){
            for(int t=0;t<N;t++){
                bool inSeeds=false;
                for(int s:seeds) if(s==t){inSeeds=true;break;}
                if(inSeeds) continue;
                vector<int> newSeeds=seeds;
                newSeeds.push_back(t);
                auto cl=computeClosure(newSeeds);
                long long sc=score(cl);
                if(sc>curScore){
                    seeds.push_back(t);
                    inCl=cl;
                    curScore=sc;
                    improved=true;
                    break;
                }
            }
        }
    }

    // Output
    cout<<seeds.size()<<"\n";
    for(int i=0;i<(int)seeds.size();i++){
        if(i) cout<<" ";
        cout<<seeds[i]+1;
    }
    cout<<"\n";
    return 0;
}