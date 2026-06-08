#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Tiny: exact problem example (D=2, N=3, K=2)
        cout << "2 3 2\n";
        cout << "1 4 2\n";
        cout << "4 1 3\n";
        cout << "2 2 1\n";

    } else if (testId == 2) {
        // Adversarial for greedy: the pair B=(1,100) + C=(100,1) is far better than any pair
        // containing A=(50,50), but greedy picks A first (best single-seed score).
        //
        // Key geometry: conv({B,C}) dominates all (x,y) with x+y<=101.
        // Adversarial diagonal: points (101-k, k) for k=2..49 and (k, 101-k) for k=2..49
        //   These have x+y=101 and are dominated by B+C but NOT by A+B or A+C.
        //   Proof (x>50 case, y<=50): A+C dom iff 49x+50y<=4950. For x=101-k, y=k:
        //     49(101-k)+50k = 4949-49k+50k = 4949+k. For k>=2: 4949+k>4949>4950 only if k>1, which is k>=2. Wait: 4949+k vs 4950: k>=2 gives 4951>=4950+1 > 4950. NOT dominated by A+C. Good.
        //
        // Medium cluster (5..47 step 3 x 5..47 step 3): x<=50,y<=50 so dominated by A alone.
        // This makes A look attractive as first greedy pick.
        //
        // After greedy picks A (score ~2251), adding B or C gives 0 gain (adversarial diagonal
        // not unlocked by A+B or A+C). Greedy stops with ~2251.
        // Optimal B+C: all medium cluster (x+y<=101) + adversarial diagonal = ~7052.

        int D = 2, K = 2;
        set<pair<int,int>> seen;
        vector<array<int,3>> pts;

        auto tryAdd = [&](int x, int y, int p) -> bool {
            if (x < 1 || y < 1) return false;
            if (!seen.insert({x, y}).second) return false;
            pts.push_back({x, y, p});
            return true;
        };

        // Seed candidates (low P so they don't look attractive alone)
        tryAdd(50, 50, 1);    // A – best single-seed score due to medium cluster
        tryAdd(1,  100, 1);   // B
        tryAdd(100, 1,  1);   // C

        // Medium cluster dominated by A=(50,50) alone: x,y in [5..47] step 3
        for (int i = 5; i <= 47; i += 3)
            for (int j = 5; j <= 47; j += 3)
                tryAdd(i, j, 10);

        // Adversarial diagonal: x+y=101, ONLY dominated by B+C
        for (int k = 2; k <= 49; k++) {
            tryAdd(101 - k, k, 100);  // x>50 side
            tryAdd(k, 101 - k, 100);  // y>50 side
        }

        int N = (int)pts.size();
        cout << D << " " << N << " " << K << "\n";
        for (auto& p : pts)
            cout << p[0] << " " << p[1] << " " << p[2] << "\n";

    } else if (testId == 3) {
        // Large random instance: D=5, N=1000, K=25. Stresses timing of LP calls.
        int D = 5, N = 1000, K = 25;
        cout << D << " " << N << " " << K << "\n";
        set<vector<int>> seen;
        int cnt = 0;
        while (cnt < N) {
            vector<int> pt(D);
            for (int d = 0; d < D; d++) pt[d] = rnd.next(1, 1000000000);
            if (seen.insert(pt).second) {
                for (int d = 0; d < D; d++) cout << pt[d] << " ";
                // Skewed: mostly negative with some large positives to create interesting structure
                int p;
                if (cnt % 10 == 0) p = rnd.next(5000000, 10000000);
                else                p = rnd.next(-10000000, 1000000);
                cout << p << "\n";
                cnt++;
            }
        }

    } else if (testId == 4) {
        // D=8 near-max: N=700, K=30. Concentrated high-value points at large coordinates
        // require specific corner seeds. Many negative-score points in middle range
        // make "obvious" high-P seeds (which cover the whole space) catastrophic.
        // Greedy may chase the highest-P individual point but activate many negatives.
        int D = 8, N = 700, K = 30;
        cout << D << " " << N << " " << K << "\n";
        set<vector<int>> seen;
        int cnt = 0;

        // Phase 1: high-value points concentrated in [900,999]^8
        while (cnt < 80) {
            vector<int> pt(D);
            for (int d = 0; d < D; d++) pt[d] = rnd.next(900, 999);
            if (seen.insert(pt).second) {
                for (int d = 0; d < D; d++) cout << pt[d] << " ";
                cout << rnd.next(1000000, 10000000) << "\n";
                cnt++;
            }
        }

        // Phase 2: negative-score points spread in [100,899]^8
        while (cnt < 500) {
            vector<int> pt(D);
            for (int d = 0; d < D; d++) pt[d] = rnd.next(100, 899);
            if (seen.insert(pt).second) {
                for (int d = 0; d < D; d++) cout << pt[d] << " ";
                cout << rnd.next(-10000000, -500000) << "\n";
                cnt++;
            }
        }

        // Phase 3: "trap" seeds – a few isolated high-P points that dominate
        // large portions including the negative cluster
        while (cnt < 560) {
            vector<int> pt(D);
            for (int d = 0; d < D; d++) pt[d] = rnd.next(850, 999);
            if (seen.insert(pt).second) {
                for (int d = 0; d < D; d++) cout << pt[d] << " ";
                cout << rnd.next(8000000, 10000000) << "\n";
                cnt++;
            }
        }

        // Phase 4: random filler to reach N=700
        while (cnt < N) {
            vector<int> pt(D);
            for (int d = 0; d < D; d++) pt[d] = rnd.next(1, 1000000000);
            if (seen.insert(pt).second) {
                for (int d = 0; d < D; d++) cout << pt[d] << " ";
                cout << rnd.next(-1000000, 1000000) << "\n";
                cnt++;
            }
        }
    }

    return 0;
}