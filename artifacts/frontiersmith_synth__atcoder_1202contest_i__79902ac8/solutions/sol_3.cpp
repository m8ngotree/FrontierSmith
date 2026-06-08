#include <bits/stdc++.h>
using namespace std;

// LP Solver: maximize c^T x s.t. Ax <= b, x >= 0
// Returns optimal value, -INF if infeasible, +INF if unbounded
const double EPS = 1e-9;
const double INF2 = 1e18;

struct LPSolver {
    int m, n;
    vector<int> N, B;
    vector<vector<double>> D;

    LPSolver(vector<vector<double>>& A, vector<double>& b, vector<double>& c)
        : m(A.size()), n(c.size()), N(n+1), B(m), D(m+2, vector<double>(n+2, 0)) {
        for(int i=0;i<m;i++) for(int j=0;j<n;j++) D[i][j]=A[i][j];
        for(int i=0;i<m;i++){B[i]=n+i;D[i][n]=-1;D[i][n+1]=b[i];}
        for(int j=0;j<n;j++){N[j]=j;D[m][j]=-c[j];}
        N[n]=-1;D[m+1][n]=1;
    }

    void pivot(int r, int s){
        double inv=1.0/D[r][s];
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
                if(s==-1||D[x][j]<D[x][s]||(fabs(D[x][j]-D[x][s])<EPS&&N[j]<N[s])) s=j;
            }
            if(D[x][s]>-EPS) return true;
            int r=-1;
            for(int i=0;i<m;i++){
                if(D[i][s]<EPS) continue;
                if(r==-1||D[i][n+1]/D[i][s]<D[r][n+1]/D[r][s]-EPS||
                   (D[i][n+1]/D[i][s]<D[r][n+1]/D[r][s]+EPS&&B[i]<B[r])) r=i;
            }
            if(r==-1) return false;
            pivot(r,s);
        }
    }

    double solve(){
        int r=0;
        for(int i=1;i<m;i++) if(D[i][n+1]<D[r][n+1]) r=i;
        if(D[r][n+1]<-EPS){
            pivot(r,n);
            if(!simplex(1)||D[m+1][n+1]<-EPS) return -INF2;
            for(int i=0;i<m;i++) if(B[i]==-1){
                int s=-1;
                for(int j=0;j<=n;j++)
                    if(s==-1||D[i][j]<D[i][s]||(fabs(D[i][j]-D[i][s])<EPS&&N[j]<N[s])) s=j;
                pivot(i,s);
            }
        }
        return simplex(2)?D[m][n+1]:-INF2;
    }
};

int D, N, K;
vector<vector<double>> X;
vector<int> P;

bool isDominated(const vector<int>& seeds, int k){
    if(seeds.empty()) return false;
    int m=seeds.size();
    // maximize t s.t. sum lambda_i*(X[s_i][c]-X[k][c]) >= t for c=0..D-1
    // sum lambda = 1, lambda >= 0
    // Variables: lambda_0..lambda_{m-1}, t+ t- (t=t+-t-)
    int n=m+2;
    vector<vector<double>> A;
    vector<double> b,c(n,0);
    c[m]=1; c[m+1]=-1;
    for(int cd=0;cd<D;cd++){
        vector<double> row(n,0);
        for(int i=0;i<m;i++) row[i]=-(X[seeds[i]][cd]-X[k][cd]);
        row[m]=1; row[m+1]=-1;
        A.push_back(row); b.push_back(0);
    }
    {vector<double> row(n,0); for(int i=0;i<m;i++) row[i]=1; A.push_back(row); b.push_back(1);}
    {vector<double> row(n,0); for(int i=0;i<m;i++) row[i]=-1; A.push_back(row); b.push_back(-1);}
    LPSolver lp(A,b,c);
    double val=lp.solve();
    return val>-EPS;
}

long long computeScore(const vector<int>& seeds){
    if(seeds.empty()) return 0;
    long long total=0;
    set<int> inSeeds(seeds.begin(),seeds.end());
    vector<bool> activated(N,false);
    for(int s:seeds) activated[s]=true;
    for(int k=0;k<N;k++){
        if(activated[k]||isDominated(seeds,k)){
            activated[k]=true;
        }
    }
    // Iterate until convergence (activation can chain)
    bool changed=true;
    while(changed){
        changed=false;
        vector<int> cur;
        for(int i=0;i<N;i++) if(activated[i]) cur.push_back(i);
        for(int k=0;k<N;k++){
            if(!activated[k]&&isDominated(cur,k)){
                activated[k]=true;
                changed=true;
            }
        }
    }
    for(int i=0;i<N;i++) if(activated[i]) total+=P[i];
    return total;
}

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cin>>D>>N>>K;
    X.resize(N,vector<double>(D));
    P.resize(N);
    for(int i=0;i<N;i++){
        for(int j=0;j<D;j++) cin>>X[i][j];
        cin>>P[i];
    }

    if(K==0){
        cout<<0<<"\n\n";
        return 0;
    }

    // Greedy: start empty, add best seed each round
    vector<int> seeds;
    vector<bool> used(N,false);
    long long curScore=0;

    for(int round=0;round<K;round++){
        int best=-1;
        long long bestScore=curScore;
        for(int i=0;i<N;i++){
            if(used[i]) continue;
            vector<int> trial=seeds;
            trial.push_back(i);
            long long sc=computeScore(trial);
            if(sc>bestScore){
                bestScore=sc;
                best=i;
            }
        }
        if(best==-1) break;
        seeds.push_back(best);
        used[best]=true;
        curScore=bestScore;
    }

    // Local search: try swapping
    bool improved=true;
    int iters=0;
    while(improved&&iters<50){
        improved=false;
        iters++;
        for(int si=0;si<(int)seeds.size();si++){
            for(int j=0;j<N;j++){
                if(used[j]) continue;
                vector<int> trial=seeds;
                trial[si]=j;
                long long sc=computeScore(trial);
                if(sc>curScore){
                    used[seeds[si]]=false;
                    used[j]=true;
                    seeds[si]=j;
                    curScore=sc;
                    improved=true;
                }
            }
        }
        // Try adding if we have room
        if((int)seeds.size()<K){
            for(int j=0;j<N;j++){
                if(used[j]) continue;
                vector<int> trial=seeds;
                trial.push_back(j);
                long long sc=computeScore(trial);
                if(sc>curScore){
                    seeds.push_back(j);
                    used[j]=true;
                    curScore=sc;
                    improved=true;
                    break;
                }
            }
        }
    }

    // Try removing seeds that don't help
    improved=true;
    while(improved){
        improved=false;
        for(int si=0;si<(int)seeds.size();si++){
            vector<int> trial;
            for(int x=0;x<(int)seeds.size();x++) if(x!=si) trial.push_back(seeds[x]);
            long long sc=computeScore(trial);
            if(sc>=curScore){
                used[seeds[si]]=false;
                seeds=trial;
                curScore=sc;
                improved=true;
                break;
            }
        }
    }

    cout<<seeds.size()<<"\n";
    for(int i=0;i<(int)seeds.size();i++){
        if(i) cout<<" ";
        cout<<seeds[i]+1;
    }
    cout<<"\n";
    return 0;
}