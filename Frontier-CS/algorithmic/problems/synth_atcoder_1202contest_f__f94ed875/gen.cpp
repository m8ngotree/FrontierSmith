#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Minimal corner case: single element, single group
        cout << "1 1 1\n";
        cout << "1\n";

    } else if (testId == 2) {
        // Small adversarial case:
        // Medians at extremes {1, 2, 9, 10}, M=15, N=15 (use all values)
        // Groups with c=1 and c=2 have NO available non-median below elements:
        //   below c=1: nothing; below c=2: only {1} which is a median.
        // So those groups can ONLY grow upward (even sizes).
        // Groups c=9 and c=10 compete for {3..8} below and {11..15} above.
        // This stresses below/above capacity balance in all solutions.
        cout << "15 15 4\n";
        cout << "1 2 9 10\n";

    } else if (testId == 3) {
        // Large uniform stress test: N=M=10^6, K=2*10^5
        // Medians evenly spaced at 5, 10, 15, ..., 10^6.
        // All non-median values (4 per gap) must be used.
        // Tests correctness and timing on maximum-size inputs.
        int M = 1000000, K = 200000, N = 1000000;
        cout << N << " " << M << " " << K << "\n";
        for (int i = 1; i <= K; i++) {
            cout << 5 * i;
            if (i < K) cout << " ";
        }
        cout << "\n";

    } else {
        // testId == 4
        // Large adversarial: N=M=10^6, K=2*10^5, randomly chosen medians.
        // Random structure exposes greedy strategies that assume uniform spacing.
        // Some groups will have very small or very large medians, with limited
        // below or above capacity, creating complex assignment pressure.
        int M = 1000000, K = 200000, N = 1000000;
        cout << N << " " << M << " " << K << "\n";

        // Generate K distinct random medians in [1, M] using a set.
        // Use rnd for reproducibility.
        set<int> chosen;
        while ((int)chosen.size() < K) {
            chosen.insert(rnd.next(1, M));
        }

        bool first = true;
        for (int v : chosen) {
            if (!first) cout << " ";
            cout << v;
            first = false;
        }
        cout << "\n";
    }

    return 0;
}