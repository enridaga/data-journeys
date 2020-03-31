
import matplotlib.pyplot as plt
import numpy as np
import scipy.special

def factorial(n):
    return int(scipy.special.gamma(n+1))
for k in range(8, 21): print(k, factorial(k))
### %sh
cat >dp.cc <<'EOF'
// Exact dynamic programming TSP solver (also known as Held-Karp algorithm)
// for optimizing small rolling-window segments of the tour.
// Considers the prime twist for santa 2018 competition in the optimization.
//
// Complexity: O(k^2 2^k) time, O(k 2^k) space.
//
// Usage: dp K <input.tour >output.tour
//
// K: number of nodes to permute in each tour segment.
//
// TSP file location is hardcoded in main().

#include <bits/stdc++.h>
#include <omp.h>

struct LogFatal {
  template<typename T> std::ostream& operator<<(const T &x) { return std::cerr; }
  ~LogFatal() { std::cerr << std::endl; abort(); }
};
#define CHECK(x) if (!(x)) LogFatal() << "Failed CHECK(" #x ") "

namespace santa {

// Distances and scores are stored internally as integers for speed.
typedef int64_t dist_t;

constexpr dist_t DIST_INF = INT64_MAX / 2;  // inf+inf must not overflow

typedef std::complex<double> Point;

// TSP problem data.
struct TSP {
  const std::vector<Point> coord;  // scaled
  const std::vector<char> prime;
  const int N;
  const double scale;

  TSP(const std::string& filename, const double scale = 1e12)
    : coord(ReadCities(filename, scale)),
      prime(Sieve(coord.size() + 100)),
      N(coord.size()),
      scale(scale) {}

  // Compute distance between cities i and j, scaled and rounded to int.
  inline dist_t Dist(int i, int j) const {
    return dist_t(abs(coord[i] - coord[j]));
  }

  // Is outgoing edge from this city to the next penalized if this city is
  // at the given 0-based position % 10 in the tour?
  inline bool PenalizedAtMod(int city, int mod) const {
    return !prime[city] && mod == 9;
  }

  // Score edge i->j given mod of 0-based position of city i in the tour.
  inline dist_t ScoreEdge(int i, int j, int mod) const {
    dist_t d = Dist(i, j);
    return d + (PenalizedAtMod(i, mod) ? d / 10 : 0);
  }

  static std::vector<char> Sieve(int n) {
    std::vector<char> res(n, 1);
    if (n >= 1) res[0] = 0;
    if (n >= 2) res[1] = 0;
    for (int i = 2; i < n; i++) {
      if (res[i]) {
        for (int j = 2 * i; j < n; j += i) res[j] = 0;
      }
    }
    return res;
  }

  // Reads cities coordinates in TSPLIB or CSV format.
  static std::vector<Point> ReadCities(const std::string& filename, double scale) {
    FILE *fp = fopen(filename.c_str(), "r");
    CHECK(fp) << "Can't open " << filename;

    char buf[100];
    fgets(buf, sizeof(buf), fp);

    const char* fmt = "%d,%lf,%lf";
    int base = 0;
    if (memcmp(buf, "NAME", 4) == 0) {  // TSPLIB
      while (fgets(buf, sizeof(buf), fp) && memcmp(buf, "NODE_COORD_SECTION", 18) != 0);
      fmt = "%d %lf %lf";
      base = 1;
    }

    std::vector<Point> res;
    int i;
    double x, y;
    while (fscanf(fp, fmt, &i, &x, &y) == 3) {
      CHECK(i == base + (int)res.size()) << "Out of order entry in " << filename;
      res.emplace_back(x * scale, y * scale);
    }

    fclose(fp);
    return res;
  }
};

// ArrayTour: a basic array-based tour implementation.
// Holds reference to problem data and array of city indices.
struct ArrayTour {
  const TSP* tsp;
  const int N;
  std::vector<int> tour_;  // including trailing 0

  ArrayTour(const ArrayTour& other)
      : tsp(other.tsp), N(other.N), tour_(other.ToVector()) {}
  ArrayTour(const char* filename, const TSP* tsp)
      : tsp(tsp), N(tsp->N), tour_(ReadVec(filename, tsp->N)) {}

  ArrayTour& operator=(const ArrayTour& other) {
    tour_ = other.tour_;
    return *this;
  }

  std::vector<int> ToVector() const { return tour_; }
  inline int operator[](int pos) const { return tour_[pos]; }
  inline int& operator[](int pos) { return tour_[pos]; }

  // Scores the whole tour, returns scaled integer distance.
  dist_t Score() const {
    dist_t res = 0;
    for (int i = 0; i < N; i++) {
      res += tsp->ScoreEdge(tour_[i], tour_[i+1], i % 10);
    }
    return res;
  }

  // Returns score in original units.
  double ScoreF() const { return Score() / tsp->scale; }

  // Write tour to file in CSV format for submission.
  void ToCSV(const char* filename = "/dev/stdout") const {
    FILE *fp = fopen(filename, "w");
    CHECK(fp) << "Can't open " << filename << " for writing";
    fprintf(fp, "Path\n");
    for (int x : ToVector()) {
      fprintf(fp, "%d\n", x);
    }
    fclose(fp);
  }

  // Reads tour from a file in CSV, TSPLIB or linkern's -S format.
  // Returns array always with the first city repeated at the end.
  // If N is specified, verifies that tour has that size.
  static std::vector<int> ReadVec(const char* filename, int N = -1) {
    std::vector<int> res;
    if (N > 0) res.reserve(N + 1);

    FILE *fp = fopen(filename, "r");
    CHECK(fp) << "Can't open " << filename;

    char buf[100];
    fgets(buf, sizeof(buf), fp);

    bool tsplib = false;
    if (memcmp(buf, "NAME", 4) == 0) {  // TSPLIB
      tsplib = true;
      while (fgets(buf, sizeof(buf), fp) && memcmp(buf, "TOUR_SECTION", 12) != 0);
    }

    int x;
    while (fscanf(fp, "%d", &x) == 1 && x != -1) {
      if (tsplib) {
        if (res.size() == 0 && x == 0) tsplib = false;
        else x--;
      }
      if (x == 0 && res.size() > 0) continue;
      res.push_back(x);
    }
    fclose(fp);

    if (N < 0) N = res.size();

    CHECK((int)res.size() == N)
        << filename << ": bad tour length " << res.size() << ", expected " << N;
    CHECK(res[0] == 0) << filename << ": tour must start with 0";

    std::vector<bool> seen(N);
    for (int x : res) {
      CHECK(0 <= x && x < N) << filename << ": index out of bounds: " << x;
      CHECK(!seen[x]) << filename << ": repeated city: " << x;
      seen[x] = true;
    }

    res.push_back(0);
    return res;
  }
};
typedef ArrayTour Tour;

// The main DP solver class.
class DPSolver {
 public:
  enum { MAXK = 24 };

  explicit DPSolver(int K, bool quiet = false) : K(K), quiet(quiet) {}

 private:
  int K;           // Number of nodes to permute
  bool quiet;      // Do not pring debug information
  const TSP* tsp;  // Saved reference to TSP problem data
  int N;           // Number of cities

  // Original city indexes in the optimized segment including
  // the two fixed surrounding cities: segment[0] and segment[K+1].
  // segment[1..K] are the nodes that we're trying to permute.
  int segment[MAXK+2];

  // Segment's positions in the tour mod 10.
  int segment_mod[MAXK+2];

  // Distance matrix between segment[] cities.
  dist_t dist[MAXK+2][MAXK+2];

  // DP memoization table
  // [bitmask of unvisited nodes][last visited node index]
  dist_t dp[1 << MAXK][MAXK];

  // Position of segment[0] in the tour (fixed node to the left of the segment)
  int left_idx_;

  dist_t orig_total;       // Original score of the segment
  dist_t best_total;       // Best found optimized score of the segment
  std::vector<int> perm;   // Reconstructed permutation of nodes

 public:
  // Optimizes a segment of the tour: tour[left_idx+1 .. left_idx+K].
  // left_idx is index of the fixed node immediately to the left of
  // optimized segment. Out of bounds indexes get wrapped around.
  // Returns true if improvement is possible. Call Apply() to commit it.
  bool CanOptimize(const ArrayTour& tour, int left_idx) {
    tsp = tour.tsp;
    N = tsp->N;
    left_idx = (left_idx % N + N) % N;  // wrap around to [0, N) range
    left_idx_ = left_idx;

    for (int i = 0; i < K+2; i++) {
      segment[i] = tour[(left_idx_ + i) % N];
      segment_mod[i] = ((left_idx_ + i) % N) % 10;
    }

    for (int i = 0; i < K+2; i++) {
      for (int j = i + 1; j < K+2; j++) {
        dist[i][j] = dist[j][i] = tsp->Dist(segment[i], segment[j]);
      }
      dist[i][i] = 0;
    }

    // Boundary case: all nodes in the segment are visited (mask=0),
    // with given index of the node at which we stopped last.
    for (int last = 0; last < K; last++) {
      dp[0][last] = tsp->ScoreEdge(segment[last+1], segment[K+1], segment_mod[K]);
    }

    // Optimize every subproblem in the order of increasing number of
    // unvisited cities, up to mask of all 1's which corresponds to
    // the full problem.
    for (int mask = 1; mask < (1 << K); mask++) {
      for (int last = 0; last < K; last++) {
        if ((mask & (1 << last)) != 0) continue;
        dp[mask][last] = DoSubproblem<false>(mask, last);
      }
    }

    orig_total = 0;
    for (int i = 0; i <= K; i++) {
      orig_total += tsp->ScoreEdge(segment[i], segment[i+1], segment_mod[i]);
    }

    best_total = DoSubproblem<false>((1 << K) - 1, -1);
    return best_total < orig_total;
  }

  // Solve a subproblem.
  //   * mask: bitmask of still unvisited nodes (segment[1..K])
  //   * last: index of last visited node
  //     Indexing in both mask and last: 0=segment[1], ..., K=1=segment[K].
  //     Special case: last=-1 refers to segment[0] for the starting case.
  //   * reconstruct: template flag whether we're reconstructing solution
  //     into perm array. Ain't nobody got time to reimplement this code
  //     twice, so we're doing it all in same method.
  template<bool reconstruct>
  inline dist_t DoSubproblem(int mask, int last) {
    if (reconstruct) {
      if (last != -1) perm.push_back(last + 1);
      if (mask == 0) return 0;
    }

    // Position of last city in the segment[] array, infer from counting bits in mask.
    int last_pos = K - __builtin_popcount(mask);

    // Is the edge from last city going to be penalized?
    bool penalized = tsp->PenalizedAtMod(segment[last+1], segment_mod[last_pos]);

    // Try every possible next city, pick one leading to least final cost.
    // Repeat twice when reconstructing solution to get the argmin.
    dist_t best = DIST_INF;
    for (int cycle = 0; cycle < (reconstruct ? 2 : 1); cycle++) {
      for (int next = 0; next < K; next++) {
        if ((mask & (1 << next)) == 0) continue;

        // Fix city 0 at its original position.
        if (segment[next+1] == 0 && left_idx_ + last_pos != N - 1) continue;

        dist_t cost = dp[mask ^ (1 << next)][next] + dist[last+1][next+1] + (penalized ? dist[last+1][next+1]/10 : 0);
        if (cost < best) {
          best = cost;
        }

        if (reconstruct && cycle == 1 && cost == best) {
          return DoSubproblem<true>(mask ^ (1 << next), next);
        }
      }
    }

    CHECK(!reconstruct) << "reconstruct fail";
    return best;
  }

  // Applies the best change found by CanOptimize() to the given tour.
  // The tour may be a different object than the one given to CanOptimize(),
  // but the segment that was optimized must be the same and at same position.
  void Apply(ArrayTour* tour) {
    perm.clear();
    DoSubproblem<true>((1 << K) - 1, -1);

    for (int i = 0; i < K; i++) {
      int idx = (left_idx_ + 1 + i) % N;
      (*tour)[idx] = segment[perm[i]];
    }

    dist_t new_total = 0;
    for (int i = 0; i <= K; i++) {
      int idx = (left_idx_ + i) % N;
      int next_idx = (idx + 1) % N;
      new_total += tsp->ScoreEdge((*tour)[idx], (*tour)[next_idx], idx % 10);
    }

    CHECK(abs(new_total - best_total) == 0)
        << "Reconstruction error: " << new_total << " vs " << best_total;

    if (!quiet) {
      fprintf(stderr, "\r%.2f: %6d %5.2f  ", tour->ScoreF(), left_idx_, (orig_total - new_total)/tsp->scale);
      for (int i : perm) fprintf(stderr, " %d", i);
      fprintf(stderr, "\n");
    }
  }
};

}  // namespace santa

int main(int argc, char** argv) {
  using namespace santa;

  if (argc <= 1) {
    fprintf(stderr, "Usage: dp K <input.tour >output.tour\n");
    return 1;
  }

  TSP tsp("../input/traveling-santa-2018-prime-paths/cities.csv");
  Tour tour("/dev/stdin", &tsp);

  int nthreads = omp_get_max_threads();
  int K = atoi(argv[1]);
  fprintf(stderr, "Initial score: %.2f, K=%d, nthreads=%d\n", tour.ScoreF(), K, nthreads);

  CHECK(2 <= K && K <= DPSolver::MAXK) << "MAXK=" << DPSolver::MAXK;

  std::vector<std::unique_ptr<DPSolver>> solvers;
  for (int i = 0; i < nthreads; i++) {
    solvers.emplace_back(new DPSolver(K));
  }

  // Try all rolling windows with a bit of overlap around the starting city.
  for (int base_pos = -25; base_pos <= tsp.N + 25; base_pos += nthreads) {
    // Uncomment if you'd like to see some progress in the console. This doesn't work well in kaggle kernels.
    // fprintf(stderr, "\r[%d]", base_pos);

    std::vector<int> success(nthreads);

    // Check nthreads rolling windows in parallel for optimization possibilities.
#pragma omp parallel for
    for (int i = 0; i < nthreads; i++) {
      success[i] = solvers[i]->CanOptimize(tour, base_pos + i);
    }

    // Apply found improvements in non-overlapping windows.
    int first_pos = INT_MAX;
    for (int i = 0; i < nthreads; i++) {
      if (success[i]) {
        solvers[i]->Apply(&tour);
        first_pos = std::min(first_pos, base_pos + i);
        i += K;
      }
    }

    // On success, roll the window back a bit for the next run to see if
    // we unlocked further improvements in a previously processed area.
    if (first_pos < INT_MAX) {
      base_pos = std::max(-25, first_pos - K - 1) - nthreads;
    }
  }

  fprintf(stderr, "Final score: %.2f\n", tour.ScoreF());

  tour.ToCSV();
}
EOF
### g++ --std=c++17 -Wall -g -Ofast -mtune=native -march=native -fopenmp -DNDEBUG -o dp dp.cc
### %bash
# K=12: 1515579.33 real 1m33s
# K=13: 1515578.73 real 3m31s
# K=14: 1515577.89 real 7m51s
# ...
# K=18: 1515572.48 ca. 200 minutes
time ./dp 18 <../input/pseudo-k-opt/1515581.355713272.csv >submission.csv