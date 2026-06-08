#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    // Read input
    long long N = inf.readLong();
    long long M = inf.readLong();
    long long K = inf.readLong();

    vector<long long> c(K);
    for (int i = 0; i < K; i++) c[i] = inf.readLong();

    // Read participant output: sizes
    vector<long long> sz(K);
    for (int i = 0; i < K; i++) sz[i] = ouf.readLong();

    long long total = 0;
    for (int i = 0; i < K; i++) {
        if (sz[i] <= 0) quitf(_wa, "Set %d has non-positive size %lld", i+1, sz[i]);
        total += sz[i];
    }
    if (total != N) quitf(_wa, "Total size %lld != N=%lld", total, N);

    vector<bool> used(M + 2, false);
    long long sub_sum = 0;

    // Read sets and validate
    for (int i = 0; i < K; i++) {
        vector<long long> s(sz[i]);
        for (long long j = 0; j < sz[i]; j++) {
            s[j] = ouf.readLong();
            if (s[j] < 1 || s[j] > M)
                quitf(_wa, "Element %lld in set %d out of range [1,%lld]", s[j], i+1, M);
            if (j > 0 && s[j] <= s[j-1])
                quitf(_wa, "Set %d not strictly increasing at position %lld", i+1, j+1);
            if (used[s[j]])
                quitf(_wa, "Element %lld appears multiple times", s[j]);
            used[s[j]] = true;
            sub_sum += s[j];
        }
        // Check median: ceil(sz[i]/2)-th smallest, 0-indexed = (sz[i]-1)/2
        long long med_idx = (sz[i] - 1) / 2;
        if (s[med_idx] != c[i])
            quitf(_wa, "Median of set %d is %lld, expected %lld", i+1, s[med_idx], c[i]);
    }

    if (!ouf.seekEof()) quitf(_wa, "Extra output after the last set");

    // Compute UB = sum of N largest values in {1,...,M}
    // = M + (M-1) + ... + (M-N+1) = N*M - N*(N-1)/2
    long long UB = N * M - N * (N - 1) / 2;

    // Compute baseline via greedy
    // State: can_right (a==b, can absorb v > c_j), can_left (a==b+1, can absorb v < c_j)
    // Sets ordered by (c_value, index); c values are distinct
    vector<bool> is_median(M + 2, false);
    for (int i = 0; i < K; i++) is_median[c[i]] = true;

    // can_right_set: groups where a_j == b_j -> can accept v > c_j
    // can_left_set:  groups where a_j == b_j+1 -> can accept v < c_j
    set<pair<long long,int>> can_right_set, can_left_set;

    long long baseline_sum = 0;
    for (int i = 0; i < K; i++) {
        baseline_sum += c[i];
        can_right_set.insert({c[i], i});
    }

    long long count = K;

    for (long long v = M; v >= 1 && count < N; v--) {
        if (is_median[v]) continue;

        long long best_c = -1;
        int best_i = -1;
        bool best_is_right = false;

        // From can_right: need c_j < v, pick largest c_j
        if (!can_right_set.empty()) {
            auto it = can_right_set.lower_bound({v, INT_MIN});
            if (it != can_right_set.begin()) {
                --it;
                if (it->first < v) {
                    if (it->first > best_c) {
                        best_c = it->first;
                        best_i = it->second;
                        best_is_right = true;
                    }
                }
            }
        }

        // From can_left: need c_j > v, pick largest c_j
        if (!can_left_set.empty()) {
            auto it = prev(can_left_set.end());
            if (it->first > v && it->first > best_c) {
                best_c = it->first;
                best_i = it->second;
                best_is_right = false;
            }
        }

        if (best_i == -1) continue; // no group can absorb v, skip

        baseline_sum += v;
        count++;

        if (best_is_right) {
            can_right_set.erase({c[best_i], best_i});
            can_left_set.insert({c[best_i], best_i});
        } else {
            can_left_set.erase({c[best_i], best_i});
            can_right_set.insert({c[best_i], best_i});
        }
    }

    // Compute normalized score
    // score = (SUM - baseline) / (UB - baseline), clamped to [0,1]
    double num = (double)(sub_sum - baseline_sum);
    double den = (double)(UB - baseline_sum);
    double ratio = 0.0;
    if (den > 0.0) {
        ratio = num / den;
    } else {
        // den == 0 means baseline == UB, any feasible solution matches upper bound
        ratio = (num >= 0.0) ? 1.0 : 0.0;
    }
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;

    quitp(ratio, "OK SUM=%lld baseline=%lld UB=%lld Ratio: %.6f",
          sub_sum, baseline_sum, UB, ratio);

    return 0;
}