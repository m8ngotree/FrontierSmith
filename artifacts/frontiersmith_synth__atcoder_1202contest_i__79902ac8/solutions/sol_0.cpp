#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double ld;

// Simplex LP solver for: min c^T x s.t. Ax <= b, x >= 0
// Returns +inf if infeasible, otherwise optimal value
// Variables indexed 0..n-1, constraints 0..m-1
struct LP {
    int m, n;
    vector<int> N, B;
    vector<vector<ld>> D2;
    LP(vector<vector<ld>>& A, vector<ld>& b, vector<ld>& c) :
        m(A.size()), n(c.size()), N(n+1), B(m), D2(m+2, vector<ld>(n+2)) {
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) D2[i][j]=A[i][j];
        for(int i=0;i<m;i++){B[i]=n+i; D2[i][n]=−1; D2[i][n+1]=b[i];}
        for(int j=0;j<n;j++){N[j]=j; D2[m][j]=−c[j];}
        N[n]=−1; D2[m+1][n]=1;
    }
    void pivot(int r, int s) {
        ld inv = 1.0/D2[r][s];
        for(int i=0;i<=m+1;i++) if(i!=r){
            for(int j=0;j<=n+1;j++) if(j!=s)
                D2[i][j] -= D2[i][s]*D2[r][j]*inv;
            D2[i][s] *= -inv;
        }
        for(int j=0;j<=n+1;j++) if(j!=s) D2[r][j]*=inv;
        D2[r][s]=inv;
        swap(B[r],N[s]);
    }
    bool simplex(int phase) {
        int x = phase==1?m+1:m;
        while(true){
            int s=-1;
            for(int j=0;j<=n;j++){
                if(phase==2&&N[j]==-1) continue;
                if(s==-1||D2[x][j]<D2[x][s]||(D2[x][j]==D2[x][s]&&N[j]<N[s])) s=j;
            }
            if(D2[x][s]>=-1e-9) return true;
            int r=-1;
            for(int i=0;i<m;i++){
                if(D2[i][s]<=1e-9) continue;
                if(r==-1||D2[i][n+1]/D2[i][s]<D2[r][n+1]/D2[r][s]||
                   (D2[i][n+1]/D2[i][s]==D2[r][n+1]/D2[r][s]&&B[i]<B[r])) r=i;
            }
            if(r==-1) return false;
            pivot(r,s);
        }
    }
    ld solve() {
        int r=0;
        for(int i=1;i<m;i++) if(D2[i][n+1]<D2[r][n+1]) r=i;
        if(D2[r][n+1]<-1e-9){
            pivot(r,n);
            if(!simplex(1)||D2[m+1][n+1]<-1e-9) return 1e18;
            for(int i=0;i<m;i++) if(B[i]==-1){
                int s=-1;
                for(int j=0;j<=n;j++)
                    if(s==-1||D2[i][j]<D2[i][s]||(D2[i][j]==D2[i][s]&&N[j]<N[s])) s=j;
                pivot(i,s);
            }
        }
        if(!simplex(2)) return 1e18;
        return -D2[m][n+1];
    }
};

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int D,N,K;
    cin>>D>>N>>K;
    
    vector<vector<ll>> x(N, vector<ll>(D));
    vector<ll> P(N);
    for(int i=0;i<N;i++){
        for(int c=0;c<D;c++) cin>>x[i][c];
        cin>>P[i];
    }
    
    // Check if point k is dominated by conv of activated set A
    // Returns true if dominated
    auto dominated = [&](const vector<bool>& active, int k) -> bool {
        vector<int> A;
        for(int i=0;i<N;i++) if(active[i]) A.push_back(i);
        if(A.empty()) return false;
        // Quick: any single point dominates k?
        for(int i:A){
            bool ok=true;
            for(int c=0;c<D;c++) if(x[i][c]<x[k][c]){ok=false;break;}
            if(ok) return true;
        }
        // Quick: for each coord, check if max >= x[k][c]
        for(int c=0;c<D;c++){
            bool any=false;
            for(int i:A) if(x[i][c]>=x[k][c]){any=true;break;}
            if(!any) return false;
        }
        // LP: min t s.t. sum_c mu_c*(x[i][c]-x[k][c]) <= t for all i in A
        //     sum_c mu_c = 1, mu_c >= 0
        // Variables: mu_0..mu_{D-1}, t (=mu_D)
        // Minimize t
        // Constraints:
        //   sum_c mu_c*(x[i][c]-x[k][c]) - t <= 0  for each i in A  [m constraints]
        //   sum_c mu_c <= 1  [constraint m]
        //   -sum_c mu_c <= -1  [constraint m+1]
        // Variables: mu_0..mu_{D-1}, t (index D), but t is free so substitute t = t+ - t-
        // Actually let's substitute: t = t' - M for large M... 
        // Easier: add t as free by t = tP - tN, tP,tN>=0
        // Variables: mu_0..mu_{D-1}, tP, tN (n = D+2)
        int m2 = A.size()+2;
        int n2 = D+2; // mu[0..D-1], tP, tN
        vector<vector<ld>> Amat(m2, vector<ld>(n2,0));
        vector<ld> bvec(m2,0), cvec(n2,0);
        // minimize tP - tN => cvec[D]=1, cvec[D+1]=-1
        cvec[D]=1; cvec[D+1]=-1; // min = -max(-c)
        // Wait LP above is min c^T x => set cvec
        // Constraint for each i in A: sum_c mu_c*y[i][c] - tP + tN <= 0
        for(int idx=0;idx<(int)A.size();idx++){
            int i=A[idx];
            for(int c=0;c<D;c++) Amat[idx][c]=(ld)(x[i][c]-x[k][c]);
            Amat[idx][D]=-1; Amat[idx][D+1]=1;
            bvec[idx]=0;
        }
        // sum mu_c <= 1
        for(int c=0;c<D;c++) Amat[A.size()][c]=1;
        bvec[A.size()]=1;
        // -sum mu_c <= -1
        for(int c=0;c<D;c++) Amat[A.size()+1][c]=-1;
        bvec[A.size()+1]=-1;
        
        LP lp(Amat,bvec,cvec);
        ld val=lp.solve();
        return val<=1e-7;
    };
    
    // Compute closure of given seed set
    auto closure = [&](const vector<int>& seeds) -> vector<bool> {
        vector<bool> active(N,false);
        for(int s:seeds) active[s]=true;
        bool changed=true;
        while(changed){
            changed=false;
            for(int k=0;k<N;k++){
                if(!active[k] && dominated(active,k)){
                    active[k]=true;
                    changed=true;
                }
            }
        }
        return active;
    };
    
    auto scoreOf = [&](const vector<bool>& active) -> ll {
        ll s=0;
        for(int i=0;i<N;i++) if(active[i]) s+=P[i];
        return s;
    };
    
    // Greedy: build seed set
    vector<int> seeds;
    ll bestScore = LLONG_MIN;
    
    // Try empty set first
    {
        vector<bool> emp(N,false);
        bestScore = max(0LL, scoreOf(emp));
    }
    
    // Greedy selection
    vector<bool> usedAsSeed(N,false);
    
    for(int step=0;step<K;step++){
        int bestSeed=-1;
        ll bestGain=LLONG_MIN;
        vector<bool> bestActive;
        
        for(int i=0;i<N;i++){
            if(usedAsSeed[i]) continue;
            vector<int> trial=seeds;
            trial.push_back(i);
            auto act=closure(trial);
            ll sc=scoreOf(act);
            if(sc>bestGain){
                bestGain=sc;
                bestSeed=i;
                bestActive=act;
            }
        }
        
        if(bestSeed==-1) break;
        if(bestGain<=bestScore && step>0) break; // no improvement, but try anyway if step==0
        
        seeds.push_back(bestSeed);
        usedAsSeed[bestSeed]=true;
        bestScore=bestGain;
    }
    
    // Local search: try swapping each seed with each non-seed
    bool improved=true;
    while(improved){
        improved=false;
        for(int si=0;si<(int)seeds.size();si++){
            for(int j=0;j<N;j++){
                if(usedAsSeed[j]) continue;
                vector<int> trial=seeds;
                trial[si]=j;
                auto act=closure(trial);
                ll sc=scoreOf(act);
                if(sc>bestScore){
                    bestScore=sc;
                    int old=seeds[si];
                    seeds[si]=j;
                    usedAsSeed[old]=false;
                    usedAsSeed[j]=true;
                    improved=true;
                    break;
                }
            }
            if(improved) break;
        }
    }
    
    // Also try adding more seeds if K allows and we have room
    // (greedy might have stopped early)
    
    // Output
    cout<<seeds.size()<<"\n";
    for(int i=0;i<(int)seeds.size();i++){
        if(i) cout<<" ";
        cout<<seeds[i]+1;
    }
    cout<<"\n";
    
    return 0;
}