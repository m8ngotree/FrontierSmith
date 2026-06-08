#include<bits/stdc++.h>
using namespace std;
typedef double ld;
typedef vector<ld> VD;
typedef vector<VD> VVD;
const ld EPS=1e-9;
// KACTL-style LP: maximize c'x s.t. Ax<=b, x>=0; returns -inf if infeasible, inf if unbounded
struct LPSolver{
    int m,n;vector<int>N,B;VVD D;
    LPSolver(VVD&A,VD&b,VD&c):m(A.size()),n(c.size()),N(n+1),B(m),D(m+2,VD(n+2)){
        for(int i=0;i<m;i++)for(int j=0;j<n;j++)D[i][j]=A[i][j];
        for(int i=0;i<m;i++){B[i]=n+i;D[i][n]=-1;D[i][n+1]=b[i];}
        for(int j=0;j<n;j++){N[j]=j;D[m][j]=-c[j];}N[n]=-1;D[m+1][n]=1;
    }
    void pivot(int r,int s){
        ld inv=1.0/D[r][s];
        for(int i=0;i<=m+1;i++)if(i!=r){for(int j=0;j<=n+1;j++)if(j!=s)D[i][j]-=D[i][s]*D[r][j]*inv;D[i][s]*=-inv;}
        for(int j=0;j<=n+1;j++)if(j!=s)D[r][j]*=inv;D[r][s]=inv;swap(B[r],N[s]);
    }
    bool simplex(int phase){
        int x=phase==1?m+1:m;
        for(;;){int s=-1;for(int j=0;j<=n;j++){if(phase==2&&N[j]==-1)continue;if(s==-1||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&N[j]<N[s]))s=j;}if(D[x][s]>=-EPS)return true;int r=-1;for(int i=0;i<m;i++){if(D[i][s]<=EPS)continue;if(r==-1||D[i][n+1]/D[i][s]<D[r][n+1]/D[r][s]||(D[i][n+1]/D[i][s]==D[r][n+1]/D[r][s]&&B[i]<B[r]))r=i;}if(r==-1)return false;pivot(r,s);}
    }
    ld solve(){int r=0;for(int i=1;i<m;i++)if(D[i][n+1]<D[r][n+1])r=i;if(D[r][n+1]<-EPS){pivot(r,n);if(!simplex(1)||D[m+1][n+1]<-EPS)return -1e18;for(int i=0;i<m;i++)if(B[i]==-1){int s=-1;for(int j=0;j<=n;j++)if(s==-1||D[i][j]<D[i][s]||(D[i][j]==D[i][s]&&N[j]<N[s]))s=j;pivot(i,s);}}if(!simplex(2))return 1e18;return D[m][n+1];}
};
int D,N,K;
vector<vector<ld>>X;
vector<long long>P;
bool dominated(const vector<int>&seeds,int k){
    int m=seeds.size();if(m==0)return false;
    // max t s.t. sum lambda_j x_{j,c} >= x_k[c]+t, sum lambda=1, lambda>=0
    // vars: lambda_0..lambda_{m-1}, t (t shifted by SH: t'=t+SH>=0)
    ld SH=2e9;
    // maximize t' subject to: for each c: -sum lambda x_{j,c} + t' <= SH - x_k[c]; sum lambda<=1; -sum lambda<=-1; t'<=2SH
    int nv=m+1;
    VVD A;VD b;
    for(int c=0;c<D;c++){VD row(nv,0);for(int j=0;j<m;j++)row[j]=-X[seeds[j]][c];row[m]=1;A.push_back(row);b.push_back(SH-X[k][c]);}
    {VD row(nv,0);for(int j=0;j<m;j++)row[j]=1;A.push_back(row);b.push_back(1);}
    {VD row(nv,0);for(int j=0;j<m;j++)row[j]=-1;A.push_back(row);b.push_back(-1);}
    {VD row(nv,0);row[m]=1;A.push_back(row);b.push_back(2*SH);}
    VD c(nv,0);c[m]=1;
    LPSolver lp(A,b,c);
    ld val=lp.solve();
    return val>=SH-1e-6;
}
long long closureScore(const vector<int>&seeds){
    long long tot=0;
    for(int i=0;i<N;i++){
        bool inSeed=false;for(int s:seeds)if(s==i){inSeed=true;break;}
        if(inSeed){tot+=P[i];continue;}
        if(dominated(seeds,i))tot+=P[i];
    }
    return tot;
}
int main(){
    ios::sync_with_stdio(false);cin.tie(0);
    cin>>D>>N>>K;
    X.resize(N,vector<ld>(D));P.resize(N);
    for(int i=0;i<N;i++){for(int j=0;j<D;j++)cin>>X[i][j];cin>>P[i];}
    vector<int>seeds;
    vector<bool>used(N,false);
    // Greedy
    for(int iter=0;iter<K;iter++){
        int best=-1;long long bestScore=-1e18;
        for(int i=0;i<N;i++){
            if(used[i])continue;
            vector<int>trial=seeds;trial.push_back(i);
            long long sc=closureScore(trial);
            if(sc>bestScore){bestScore=sc;best=i;}
        }
        if(best==-1)break;
        long long curScore=(seeds.empty()?0:closureScore(seeds));
        if(bestScore<=curScore&&!seeds.empty())break;
        seeds.push_back(best);used[best]=true;
    }
    // Local search: try swapping
    bool improved=true;
    int lsIter=0;
    while(improved&&lsIter<50){
        improved=false;lsIter++;
        long long cur=closureScore(seeds);
        // Try replacing each seed with each non-seed
        for(int si=0;si<(int)seeds.size()&&!improved;si++){
            for(int ni=0;ni<N&&!improved;ni++){
                if(used[ni])continue;
                vector<int>trial=seeds;trial[si]=ni;
                long long sc=closureScore(trial);
                if(sc>cur){cur=sc;seeds[si]=ni;used[ni]=true;used[ni^0]=false;// fix
                    // rebuild used
                    fill(used.begin(),used.end(),false);
                    for(int s:seeds)used[s]=true;
                    improved=true;
                }
            }
        }
        // Try adding if possible
        if(!improved&&(int)seeds.size()<K){
            for(int ni=0;ni<N&&!improved;ni++){
                if(used[ni])continue;
                vector<int>trial=seeds;trial.push_back(ni);
                long long sc=closureScore(trial);
                if(sc>cur){seeds.push_back(ni);used[ni]=true;improved=true;cur=sc;}
            }
        }
    }
    cout<<seeds.size()<<"\n";
    for(int i=0;i<(int)seeds.size();i++)cout<<seeds[i]+1<<" \n"[i+1==(int)seeds.size()];
    if(seeds.empty())cout<<"\n";
}