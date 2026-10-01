// EVOLVE-BLOCK-START
#include <bits/stdc++.h>
using namespace std;

class FastInput {
    static constexpr int BUFSIZE = 1 << 20;
    char buffer[BUFSIZE];
    int position = 0, length = 0;

public:
    // Reads the next nonnegative integer using a buffered input stream.
    int readInt() {
        char c;
        do {
            if (position == length) {
                length = (int)fread(buffer, 1, BUFSIZE, stdin);
                position = 0;
                if (length == 0) return -1;
            }
            c = buffer[position++];
        } while (c <= ' ');

        int value = 0;
        do {
            value = value * 10 + (c - '0');
            if (position == length) {
                length = (int)fread(buffer, 1, BUFSIZE, stdin);
                position = 0;
                if (length == 0) break;
            }
            c = buffer[position++];
        } while (c > ' ');
        return value;
    }
};

struct Edge {
    int u, v, x;
    uint16_t w;
};

struct ErrorEntry {
    double error;
    int index;
};

struct MedianEntry {
    double target;
    double weight;
};

int main() {
    // Tracks the CPU budget so optional diversification cannot threaten validity.
    const clock_t searchStartTime = clock();

    // Fits vertex values in log-space, refines them by exact coordinate medians,
    // repeatedly removes the D largest residuals, and falls back to baseline if needed.
    FastInput input;
    const int n = input.readInt();
    const int m = input.readInt();
    const int discardLimit = input.readInt();

    if (n < 0 || m < 0 || discardLimit < 0) return 0;

    vector<Edge> edges(m);
    vector<double> edgeLog(m);
    vector<int> degree(n, 0);

    for (int i = 0; i < m; ++i) {
        int u = input.readInt() - 1;
        int v = input.readInt() - 1;
        int x = input.readInt();
        int w = input.readInt();
        edges[i] = {u, v, x, (uint16_t)w};
        edgeLog[i] = log((double)x);
        ++degree[u];
        ++degree[v];
    }

    if (discardLimit == m) {
        for (int i = 0; i < n; ++i) {
            if (i) putchar(' ');
            putchar('1');
        }
        putchar('\n');
        printf("%d", m);
        for (int i = 1; i <= m; ++i) printf(" %d", i);
        putchar('\n');
        return 0;
    }

    vector<int> offset(n + 1, 0);
    for (int i = 0; i < n; ++i) offset[i + 1] = offset[i] + degree[i];

    vector<int> adjacency(2LL * m);
    vector<int> cursor = offset;
    int maximumDegree = 0;
    for (int i = 0; i < n; ++i) maximumDegree = max(maximumDegree, degree[i]);

    for (int i = 0; i < m; ++i) {
        adjacency[cursor[edges[i].u]++] = i;
        adjacency[cursor[edges[i].v]++] = i;
    }

    double weightedLogSum = 0.0;
    double weightSum = 0.0;
    for (int i = 0; i < m; ++i) {
        weightedLogSum += (double)edges[i].w * edgeLog[i];
        weightSum += edges[i].w;
    }

    const double LOG_LIMIT = log(1e9);
    double initialLog = weightSum > 0.0 ? 0.5 * weightedLogSum / weightSum : 0.0;
    initialLog = min(LOG_LIMIT, max(0.0, initialLog));

    vector<double> logs(n, initialLog), nextLogs(n, initialLog);

    // Robust log-space initialization. A larger available discard fraction
    // permits a sharper IRLS influence function, while D=0 preserves the
    // original conservative residual floor.
    const double discardFraction =
        min(1.0, (double)discardLimit / (double)m);
    const double initialRobustFloor =
        0.20 - 0.12 * discardFraction;

    // Run relaxed robust Jacobi fitting. Half-relaxation suppresses the
    // period-two oscillation induced by the sum equations on bipartite graphs.
    for (int iteration = 0; iteration < 30; ++iteration) {
        for (int u = 0; u < n; ++u) {
            if (degree[u] == 0) {
                nextLogs[u] = 0.0;
                continue;
            }

            double numerator = 0.0;
            double denominator = 0.0;
            for (int p = offset[u]; p < offset[u + 1]; ++p) {
                int edgeIndex = adjacency[p];
                const Edge& e = edges[edgeIndex];
                int v = e.u ^ e.v ^ u;
                double lx = edgeLog[edgeIndex];
                double residual = logs[u] + logs[v] - lx;
                double robustWeight =
                    (double)e.w / max(initialRobustFloor, abs(residual));
                numerator += robustWeight * (lx - logs[v]);
                denominator += robustWeight;
            }

            double candidate = denominator > 0.0 ? numerator / denominator : logs[u];
            candidate = 0.5 * logs[u] + 0.5 * candidate;
            nextLogs[u] = min(LOG_LIMIT, max(0.0, candidate));
        }
        logs.swap(nextLogs);
    }

    vector<int> values(n, 1);
    for (int i = 0; i < n; ++i) {
        long long candidate = llround(exp(logs[i]));
        candidate = max(1LL, min(1000000000LL, candidate));
        values[i] = (int)candidate;
    }

    // Preserve the robust log-space initialization for an independent
    // early-trimming basin evaluated after the original fitting path.
    const vector<int> robustInitialValues = values;

    vector<char> discarded(m, 0);
    vector<MedianEntry> medianEntries;
    medianEntries.reserve(maximumDegree);

    auto coordinatePass = [&](bool reverseOrder, bool ignoreDiscarded,
                              const vector<int>* customOrder = nullptr,
                              double discardedPenalty = 0.0,
                              const vector<float>* discardedWeights = nullptr,
                              const vector<int>* neighborSnapshot = nullptr) {
        // Minimizes every integer coordinate by a weighted median. An optional
        // snapshot makes the sweep synchronous, supplying an order-independent
        // Jacobi restart; otherwise the pass uses Gauss-Seidel updates.
        for (int step = 0; step < n; ++step) {
            int position = reverseOrder ? n - 1 - step : step;
            int u = customOrder ? (*customOrder)[position] : position;
            medianEntries.clear();
            double totalMedianWeight = 0.0;

            for (int p = offset[u]; p < offset[u + 1]; ++p) {
                int edgeIndex = adjacency[p];
                // A supplied continuation vector defines the influence of
                // every observation. Otherwise, use ordinary hard trimming.
                double edgeScale = 1.0;
                if (discardedWeights) {
                    edgeScale = (*discardedWeights)[edgeIndex];
                } else if (ignoreDiscarded && discarded[edgeIndex]) {
                    edgeScale = discardedPenalty;
                }
                if (edgeScale == 0.0) continue;

                const Edge& e = edges[edgeIndex];
                int v = e.u ^ e.v ^ u;
                double neighbor = neighborSnapshot
                                ? (*neighborSnapshot)[v]
                                : values[v];
                double target = (double)e.x / neighbor;
                double coefficient =
                    edgeScale * (double)e.w * neighbor / (double)e.x;
                medianEntries.push_back({target, coefficient});
                totalMedianWeight += coefficient;
            }

            if (medianEntries.empty()) continue;

            // Three-way weighted quickselect avoids sorting all incident
            // observations. Median-of-three pivots and duplicate grouping keep
            // the selection robust when many observations share one target.
            int left = 0;
            int right = (int)medianEntries.size();
            double requiredWeight = 0.5 * totalMedianWeight;
            double median = medianEntries[0].target;

            while (right - left > 1) {
                double first = medianEntries[left].target;
                double middle =
                    medianEntries[left + (right - left) / 2].target;
                double last = medianEntries[right - 1].target;
                double pivot = max(min(first, middle),
                                   min(max(first, middle), last));

                int lessEnd = left;
                int scan = left;
                int greaterBegin = right;
                while (scan < greaterBegin) {
                    if (medianEntries[scan].target < pivot) {
                        swap(medianEntries[lessEnd++], medianEntries[scan++]);
                    } else if (medianEntries[scan].target > pivot) {
                        swap(medianEntries[scan],
                             medianEntries[--greaterBegin]);
                    } else {
                        ++scan;
                    }
                }

                double lessWeight = 0.0;
                double equalWeight = 0.0;
                for (int i = left; i < lessEnd; ++i)
                    lessWeight += medianEntries[i].weight;
                for (int i = lessEnd; i < greaterBegin; ++i)
                    equalWeight += medianEntries[i].weight;

                if (requiredWeight <= lessWeight) {
                    right = lessEnd;
                } else if (requiredWeight <= lessWeight + equalWeight) {
                    median = pivot;
                    break;
                } else {
                    requiredWeight -= lessWeight + equalWeight;
                    left = greaterBegin;
                }
            }

            if (right - left == 1)
                median = medianEntries[left].target;

            long long low = (long long)floor(median);
            low = max(1LL, min(1000000000LL, low));
            long long high = min(1000000000LL, low + 1);

            // Evaluate both adjacent integers in one scan. This preserves the
            // exact coordinate objective while reducing sweep memory traffic.
            double lowLoss = 0.0;
            double highLoss = 0.0;
            for (const MedianEntry& entry : medianEntries) {
                lowLoss += entry.weight *
                           abs((double)low - entry.target);
                highLoss += entry.weight *
                            abs((double)high - entry.target);
            }
            values[u] = (int)(highLoss < lowLoss ? high : low);
        }
    };

    vector<ErrorEntry> errors(m);
    auto selectDiscarded = [&](bool baselineValues, int trimCount = -1) {
        // Selects the requested number of largest residuals; -1 means the full D.
        if (trimCount < 0) trimCount = discardLimit;
        trimCount = min(trimCount, discardLimit);

        double retainedLoss = 0.0;
        for (int i = 0; i < m; ++i) {
            const Edge& e = edges[i];
            long long product = baselineValues
                              ? 1LL
                              : (long long)values[e.u] * values[e.v];
            double error = (double)e.w *
                           (double)llabs(product - (long long)e.x) / (double)e.x;
            errors[i] = {error, i};
            retainedLoss += error;
        }

        const double totalLoss = retainedLoss;

        if (trimCount > 0) {
            nth_element(errors.begin(), errors.begin() + trimCount, errors.end(),
                        [](const ErrorEntry& a, const ErrorEntry& b) {
                            if (a.error != b.error) return a.error > b.error;
                            return a.index < b.index;
                        });
        }

        fill(discarded.begin(), discarded.end(), 0);
        for (int i = 0; i < trimCount; ++i) {
            discarded[errors[i].index] = 1;
            retainedLoss -= errors[i].error;
        }

        // If discarded outliers nearly equal the entire sum, subtraction can
        // lose all retained-loss precision. Re-sum only in that rare regime.
        if (totalLoss > 0.0 && retainedLoss <= totalLoss * 1e-10) {
            long double accurateLoss = 0.0L;
            for (const ErrorEntry& entry : errors)
                if (!discarded[entry.index])
                    accurateLoss += (long double)entry.error;
            retainedLoss = (double)accurateLoss;
        }
        return max(0.0, retainedLoss);
    };

    // Candidate A follows the original approach: first fit every observation,
    // then alternate retained-coordinate minimization and optimal trimming.
    coordinatePass(false, false);
    coordinatePass(true, false);
    selectDiscarded(false);

    // Alternate exact coordinate minimization and optimal trimming, retaining
    // the final iteration's already-computed loss.
    double fittedLoss = 0.0;
    for (int iteration = 0; iteration < 7; ++iteration) {
        vector<int> previousValues = values;
        coordinatePass(iteration & 1, true);
        coordinatePass(!(iteration & 1), true);
        fittedLoss = selectDiscarded(false);
        // Deterministic coordinate descent and trimming cannot change again
        // after a complete pair of sweeps leaves every coordinate unchanged.
        if (values == previousValues) break;
    }
    vector<int> bestValues = values;
    vector<char> bestDiscarded = discarded;

    // Exact-product and diversified restarts are also valuable when D=0:
    // clean sparse instances can often be reconstructed exactly from a seed.
    if (discardLimit > 0 || m <= 250000) {
        // Exact-product propagation restart. Seed both sides of the trimming
        // boundary, choose the nearest exact factorization, then spread exact
        // quotients through the most reliable retained observations.
        if (m <= 250000 &&
            (double)(clock() - searchStartTime) / CLOCKS_PER_SEC < 7.60) {
            constexpr int SEEDS_PER_CLASS = 6;
            priority_queue<pair<double, int>> marginalQueue;
            priority_queue<pair<double, int>,
                           vector<pair<double, int>>,
                           greater<pair<double, int>>> retainedQueue;

            for (int i = 0; i < m; ++i) {
                const Edge& e = edges[i];
                long long product =
                    (long long)bestValues[e.u] * bestValues[e.v];
                double error = (double)e.w *
                    (double)llabs(product - (long long)e.x) / (double)e.x;
                pair<double, int> item = {error, i};

                if (bestDiscarded[i]) {
                    if ((int)marginalQueue.size() < SEEDS_PER_CLASS) {
                        marginalQueue.push(item);
                    } else if (item < marginalQueue.top()) {
                        marginalQueue.pop();
                        marginalQueue.push(item);
                    }
                } else {
                    if ((int)retainedQueue.size() < SEEDS_PER_CLASS) {
                        retainedQueue.push(item);
                    } else if (item > retainedQueue.top()) {
                        retainedQueue.pop();
                        retainedQueue.push(item);
                    }
                }
            }

            vector<int> marginalSeeds, retainedSeeds, propagationSeeds;
            while (!marginalQueue.empty()) {
                marginalSeeds.push_back(marginalQueue.top().second);
                marginalQueue.pop();
            }
            reverse(marginalSeeds.begin(), marginalSeeds.end());

            while (!retainedQueue.empty()) {
                retainedSeeds.push_back(retainedQueue.top().second);
                retainedQueue.pop();
            }
            reverse(retainedSeeds.begin(), retainedSeeds.end());

            for (int position = 0; position < SEEDS_PER_CLASS; ++position) {
                if (position < (int)marginalSeeds.size())
                    propagationSeeds.push_back(marginalSeeds[position]);
                if (position < (int)retainedSeeds.size())
                    propagationSeeds.push_back(retainedSeeds[position]);
            }

            for (int seedIndex : propagationSeeds) {
                if ((double)(clock() - searchStartTime) /
                        CLOCKS_PER_SEC > 8.45)
                    break;

                values = bestValues;
                discarded = bestDiscarded;
                const Edge& seed = edges[seedIndex];

                int chosenU = values[seed.u];
                int chosenV = values[seed.v];
                double chosenDistance =
                    numeric_limits<double>::infinity();

                auto considerFactorization = [&](int candidateU,
                                                 int candidateV) {
                    double distance =
                        abs(log((double)candidateU /
                                (double)bestValues[seed.u])) +
                        abs(log((double)candidateV /
                                (double)bestValues[seed.v]));
                    if (distance < chosenDistance) {
                        chosenDistance = distance;
                        chosenU = candidateU;
                        chosenV = candidateV;
                    }
                };

                for (int divisor = 1;
                     (long long)divisor * divisor <= seed.x; ++divisor) {
                    if (seed.x % divisor != 0) continue;
                    int other = seed.x / divisor;
                    considerFactorization(divisor, other);
                    if (divisor != other)
                        considerFactorization(other, divisor);
                }

                values[seed.u] = chosenU;
                values[seed.v] = chosenV;

                vector<char> recovered(n, 0);
                recovered[seed.u] = recovered[seed.v] = 1;

                using FrontierEntry = tuple<double, int, int>;
                priority_queue<FrontierEntry> frontier;

                // Adds retained incident edges, prioritizing high weight and
                // compatibility with the incumbent products.
                auto pushFrontier = [&](int u) {
                    for (int p = offset[u]; p < offset[u + 1]; ++p) {
                        int edgeIndex = adjacency[p];
                        if (bestDiscarded[edgeIndex]) continue;

                        const Edge& e = edges[edgeIndex];
                        int v = e.u ^ e.v ^ u;
                        if (recovered[v]) continue;

                        long long oldProduct =
                            (long long)bestValues[e.u] *
                            bestValues[e.v];
                        double oldError =
                            (double)e.w *
                            (double)llabs(oldProduct - (long long)e.x) /
                            (double)e.x;
                        double confidence =
                            (double)e.w / (1.0 + oldError);
                        frontier.emplace(confidence, edgeIndex, u);
                    }
                };

                pushFrontier(seed.u);
                pushFrontier(seed.v);

                while (!frontier.empty()) {
                    auto [confidence, edgeIndex, from] = frontier.top();
                    frontier.pop();

                    const Edge& e = edges[edgeIndex];
                    int to = e.u ^ e.v ^ from;
                    if (!recovered[from] || recovered[to]) continue;
                    if (e.x % values[from] != 0) continue;

                    int candidate = e.x / values[from];
                    double movement = abs(log(
                        (double)candidate /
                        (double)bestValues[to]));
                    if (movement > 1.60) continue;

                    values[to] = candidate;
                    recovered[to] = 1;
                    pushFrontier(to);
                }

                selectDiscarded(false);
                double propagationLoss = 0.0;
                for (int iteration = 0; iteration < 2; ++iteration) {
                    vector<int> previousValues = values;
                    coordinatePass(iteration & 1, true);
                    coordinatePass(!(iteration & 1), true);
                    propagationLoss = selectDiscarded(false);
                    if (values == previousValues) break;
                }

                if (propagationLoss < fittedLoss) {
                    fittedLoss = propagationLoss;
                    bestValues = values;
                    bestDiscarded = discarded;
                }
            }
        }

        // Candidate B trims directly from the robust initialization, avoiding
        // the initial all-edge sweeps when a few extreme observations dominate.
        values = robustInitialValues;
        selectDiscarded(false);

        // Continue alternating exact retained-coordinate descent and optimal
        // trimming to stabilize discard membership near the local optimum.
        // Refit against the current trimmed set and reuse the last exact loss.
        double earlyTrimLoss = 0.0;
        for (int iteration = 0; iteration < 7; ++iteration) {
            vector<int> previousValues = values;
            coordinatePass(iteration & 1, true);
            coordinatePass(!(iteration & 1), true);
            earlyTrimLoss = selectDiscarded(false);
            if (values == previousValues) break;
        }
        if (earlyTrimLoss < fittedLoss) {
            fittedLoss = earlyTrimLoss;
            bestValues = values;
            bestDiscarded = discarded;
        }

        // Candidate C first trims from the robust estimate, refits the
        // continuous log model on retained observations, and only then rounds
        // and applies exact integer coordinate minimization.
        values = robustInitialValues;
        selectDiscarded(false);
        for (int u = 0; u < n; ++u) logs[u] = log((double)values[u]);

        // Alternating trimmed log-space IRLS: refit the retained graph, then
        // periodically round and reselect the exact top-D product residuals.
        for (int iteration = 0; iteration < 10; ++iteration) {
            for (int u = 0; u < n; ++u) {
                double numerator = 0.0;
                double denominator = 0.0;

                for (int p = offset[u]; p < offset[u + 1]; ++p) {
                    int edgeIndex = adjacency[p];
                    if (discarded[edgeIndex]) continue;

                    const Edge& e = edges[edgeIndex];
                    int v = e.u ^ e.v ^ u;
                    double residual = logs[u] + logs[v] - edgeLog[edgeIndex];
                    double robustWeight =
                        (double)e.w / max(0.08, abs(residual));
                    numerator += robustWeight *
                                 (edgeLog[edgeIndex] - logs[v]);
                    denominator += robustWeight;
                }

                if (denominator == 0.0) {
                    nextLogs[u] = logs[u];
                } else {
                    double candidate = numerator / denominator;
                    candidate = 0.5 * logs[u] + 0.5 * candidate;
                    nextLogs[u] = min(LOG_LIMIT, max(0.0, candidate));
                }
            }
            logs.swap(nextLogs);

            if (iteration % 3 == 2) {
                for (int u = 0; u < n; ++u) {
                    long long candidate = llround(exp(logs[u]));
                    candidate = max(1LL, min(1000000000LL, candidate));
                    values[u] = (int)candidate;
                }
                selectDiscarded(false);
            }
        }

        for (int u = 0; u < n; ++u) {
            long long candidate = llround(exp(logs[u]));
            candidate = max(1LL, min(1000000000LL, candidate));
            values[u] = (int)candidate;
        }

        selectDiscarded(false);
        // Polish the rounded log-space basin until its integer coordinates
        // stabilize. Each sweep and top-D reselection is loss non-increasing.
        double refittedLoss = 0.0;
        for (int iteration = 0; iteration < 6; ++iteration) {
            vector<int> previousValues = values;
            coordinatePass(iteration & 1, true);
            coordinatePass(!(iteration & 1), true);
            refittedLoss = selectDiscarded(false);
            if (values == previousValues) break;
        }
        if (refittedLoss < fittedLoss) {
            fittedLoss = refittedLoss;
            bestValues = values;
            bestDiscarded = discarded;
        }

        // Log-rounding ensemble: coordinated floor/ceil patterns can enter
        // integer basins that nearest rounding followed by coordinate descent
        // cannot reach, especially when the recovered factors are small.
        if (m <= 150000 &&
            (double)(clock() - searchStartTime) / CLOCKS_PER_SEC < 7.40) {
            for (int trial = 0; trial < 4; ++trial) {
                for (int u = 0; u < n; ++u) {
                    double continuous = exp(logs[u]);
                    long long low = (long long)floor(continuous);
                    long long high = (long long)ceil(continuous);

                    bool chooseHigh;
                    if (trial == 0) {
                        chooseHigh = false;
                    } else if (trial == 1) {
                        chooseHigh = true;
                    } else {
                        uint64_t z =
                            (uint64_t)(u + 1) *
                            0x9e3779b97f4a7c15ULL +
                            (uint64_t)trial *
                            0xbf58476d1ce4e5b9ULL;
                        z ^= z >> 30;
                        z *= 0xbf58476d1ce4e5b9ULL;
                        z ^= z >> 27;
                        z *= 0x94d049bb133111ebULL;
                        z ^= z >> 31;
                        chooseHigh = (z & 1ULL) != 0;
                    }

                    long long candidate = chooseHigh ? high : low;
                    values[u] = (int)max(
                        1LL, min(1000000000LL, candidate));
                }

                selectDiscarded(false);
                double roundingLoss = 0.0;
                for (int iteration = 0; iteration < 4; ++iteration) {
                    vector<int> previousValues = values;
                    coordinatePass((iteration + trial) & 1, true);
                    coordinatePass(!((iteration + trial) & 1), true);
                    roundingLoss = selectDiscarded(false);
                    if (values == previousValues) break;
                }

                if (roundingLoss < fittedLoss) {
                    fittedLoss = roundingLoss;
                    bestValues = values;
                    bestDiscarded = discarded;
                }

                if ((double)(clock() - searchStartTime) /
                        CLOCKS_PER_SEC > 8.15)
                    break;
            }
        }

        // Candidate D gradually introduces outliers instead of immediately
        // trimming all D, reducing early discard-set mistakes.
        values = robustInitialValues;
        // Increase the trimming budget gradually; stage four computes the
        // exact full-budget loss used to evaluate this candidate.
        double graduatedLoss = 0.0;
        for (int stage = 1; stage <= 4; ++stage) {
            int trimCount =
                (int)(((long long)discardLimit * stage + 3) / 4);
            selectDiscarded(false, trimCount);
            coordinatePass(stage & 1, true);
            coordinatePass(!(stage & 1), true);
            graduatedLoss = selectDiscarded(false, trimCount);
        }
        if (graduatedLoss < fittedLoss) {
            fittedLoss = graduatedLoss;
            bestValues = values;
            bestDiscarded = discarded;
        }

        // Candidate E starts from the robust estimate but uses a deterministic
        // shuffled Gauss-Seidel order to explore a distinct trimmed basin.
        vector<pair<uint64_t, int>> keyedOrder(n);
        for (int u = 0; u < n; ++u) {
            uint64_t z = (uint64_t)(u + 1) * 0x9e3779b97f4a7c15ULL;
            z ^= z >> 30;
            z *= 0xbf58476d1ce4e5b9ULL;
            z ^= z >> 27;
            z *= 0x94d049bb133111ebULL;
            z ^= z >> 31;
            keyedOrder[u] = {z, u};
        }
        sort(keyedOrder.begin(), keyedOrder.end());

        vector<int> shuffledOrder(n);
        for (int i = 0; i < n; ++i)
            shuffledOrder[i] = keyedOrder[i].second;

        values = robustInitialValues;
        selectDiscarded(false);
        double shuffledLoss = 0.0;
        for (int iteration = 0; iteration < 7; ++iteration) {
            coordinatePass(iteration & 1, true, &shuffledOrder);
            shuffledLoss = selectDiscarded(false);
        }
        if (shuffledLoss < fittedLoss) {
            fittedLoss = shuffledLoss;
            bestValues = values;
            bestDiscarded = discarded;
        }

        // Uniform soft continuation crosses discard-set boundaries, while
        // exact hard trimming ensures that only genuine improvements survive.
        const double continuationPenalties[] = {0.03, 0.12};
        for (double penalty : continuationPenalties) {
            values = bestValues;
            discarded = bestDiscarded;

            coordinatePass(false, true, nullptr, penalty);
            coordinatePass(true, true, nullptr, penalty);
            selectDiscarded(false);

            coordinatePass(false, true);
            coordinatePass(true, true);
            double continuationLoss = selectDiscarded(false);

            if (continuationLoss < fittedLoss) {
                fittedLoss = continuationLoss;
                bestValues = values;
                bestDiscarded = discarded;
            }
        }

        // Residual-aware continuation focuses on discarded observations near
        // the top-D boundary. Such edges are plausible exchange candidates;
        // much larger residuals are likely true outliers and receive little pull.
        values = bestValues;
        discarded = bestDiscarded;
        vector<float> adaptiveWeights(m, 0.0f);
        double boundaryError = numeric_limits<double>::infinity();

        for (int i = 0; i < m; ++i) {
            if (!discarded[i]) continue;
            const Edge& e = edges[i];
            long long product = (long long)values[e.u] * values[e.v];
            double error = (double)e.w *
                           (double)llabs(product - (long long)e.x) /
                           (double)e.x;
            boundaryError = min(boundaryError, error);
        }

        if (isfinite(boundaryError)) {
            boundaryError = max(boundaryError, 1e-15);

            // Smooth the hard top-D boundary with a steep logistic influence
            // function. Edges clearly below the boundary retain nearly full
            // weight, clear outliers receive almost none, and observations on
            // either side of the boundary can exchange membership.
            for (int i = 0; i < m; ++i) {
                const Edge& e = edges[i];
                long long product = (long long)values[e.u] * values[e.v];
                double error = (double)e.w *
                               (double)llabs(product - (long long)e.x) /
                               (double)e.x;
                double ratio = error / boundaryError;
                double exponent =
                    max(-20.0, min(20.0, 12.0 * (ratio - 1.0)));
                double influence = 1.0 / (1.0 + exp(exponent));
                adaptiveWeights[i] =
                    (float)max(0.005, min(0.995, influence));
            }

            coordinatePass(false, true, nullptr, 0.0, &adaptiveWeights);
            coordinatePass(true, true, nullptr, 0.0, &adaptiveWeights);
            selectDiscarded(false);

            coordinatePass(false, true);
            coordinatePass(true, true);
            double adaptiveLoss = selectDiscarded(false);

            if (adaptiveLoss < fittedLoss) {
                fittedLoss = adaptiveLoss;
                bestValues = values;
                bestDiscarded = discarded;
            }
        }

        // Large-neighborhood discard search: reintroduce marginal outliers,
        // exchange both sides of the trim boundary, and retain hashed basins.
        for (int trial = 0; trial < 4; ++trial) {
            values = bestValues;
            discarded = bestDiscarded;
            bool reverseFirst = trial & 1;

            if (trial < 2) {
                double boundary = numeric_limits<double>::infinity();
                for (int i = 0; i < m; ++i) {
                    if (!discarded[i]) continue;
                    const Edge& e = edges[i];
                    long long product =
                        (long long)values[e.u] * values[e.v];
                    double error = (double)e.w *
                                   (double)llabs(product - (long long)e.x) /
                                   (double)e.x;
                    boundary = min(boundary, error);
                }

                if (isfinite(boundary)) {
                    double upper = boundary * 1.25 + 1e-12;
                    double lower = boundary * 0.80;
                    for (int i = 0; i < m; ++i) {
                        const Edge& e = edges[i];
                        long long product =
                            (long long)values[e.u] * values[e.v];
                        double error = (double)e.w *
                                       (double)llabs(product - (long long)e.x) /
                                       (double)e.x;
                        if (discarded[i] && error <= upper) {
                            discarded[i] = 0;
                        } else if (trial == 1 && boundary > 1e-12 &&
                                   !discarded[i] && error >= lower) {
                            discarded[i] = 1;
                        }
                    }
                }
            } else {
                int hashedTrial = trial - 2;
                uint64_t mask = hashedTrial == 0 ? 15ULL : 3ULL;
                for (int i = 0; i < m; ++i) {
                    if (!discarded[i]) continue;
                    uint64_t z = (uint64_t)(i + 1) ^
                                 (0x9e3779b97f4a7c15ULL *
                                  (hashedTrial + 1));
                    z ^= z >> 30;
                    z *= 0xbf58476d1ce4e5b9ULL;
                    z ^= z >> 27;
                    z *= 0x94d049bb133111ebULL;
                    z ^= z >> 31;
                    if ((z & mask) == 0) discarded[i] = 0;
                }
            }

            coordinatePass(reverseFirst, true);
            coordinatePass(!reverseFirst, true);
            selectDiscarded(false);

            coordinatePass(!reverseFirst, true);
            coordinatePass(reverseFirst, true);
            double dropoutLoss = selectDiscarded(false);

            if (dropoutLoss < fittedLoss) {
                fittedLoss = dropoutLoss;
                bestValues = values;
                bestDiscarded = discarded;
            }
        }

        // Bipartite-gauge basin: products are invariant in exact arithmetic
        // when one partition is scaled and the other inversely scaled. Integer
        // rounding perturbs that gauge and can expose a better factorization.
        values = bestValues;
        discarded = bestDiscarded;

        vector<int> bipartiteColor(n, -1);
        vector<int> component(n, -1);
        vector<char> badComponent;
        vector<int> queue;
        queue.reserve(n);
        int componentCount = 0;

        for (int start = 0; start < n; ++start) {
            if (bipartiteColor[start] != -1) continue;

            badComponent.push_back(0);
            queue.clear();
            queue.push_back(start);
            bipartiteColor[start] = 0;
            component[start] = componentCount;

            for (size_t head = 0; head < queue.size(); ++head) {
                int u = queue[head];
                for (int p = offset[u]; p < offset[u + 1]; ++p) {
                    int edgeIndex = adjacency[p];
                    if (discarded[edgeIndex]) continue;

                    const Edge& e = edges[edgeIndex];
                    int v = e.u ^ e.v ^ u;
                    if (bipartiteColor[v] == -1) {
                        bipartiteColor[v] = bipartiteColor[u] ^ 1;
                        component[v] = componentCount;
                        queue.push_back(v);
                    } else if (bipartiteColor[v] == bipartiteColor[u]) {
                        badComponent[componentCount] = 1;
                    }
                }
            }
            ++componentCount;
        }

        // Search integer gauge scales on bipartite components. Multiplying one
        // side and inversely scaling the other preserves continuous products,
        // while different rounding patterns can reach distinct integer basins.
        for (int gaugeTrial = 0; gaugeTrial < 6; ++gaugeTrial) {
            if (gaugeTrial >= 2) {
                double elapsed =
                    (double)(clock() - searchStartTime) / CLOCKS_PER_SEC;
                if (m > 250000 || elapsed > 8.35) break;
            }

            values = bestValues;
            discarded = bestDiscarded;

            int scale = gaugeTrial < 2 ? 2 :
                        gaugeTrial < 4 ? 3 : 5;
            int scaledColor = gaugeTrial & 1;

            for (int u = 0; u < n; ++u) {
                if (badComponent[component[u]]) continue;
                bool multiplySide =
                    bipartiteColor[u] == scaledColor;
                long long candidate = multiplySide
                    ? (long long)scale * values[u]
                    : ((long long)values[u] + scale / 2) / scale;
                values[u] = (int)max(
                    1LL, min(1000000000LL, candidate));
            }

            selectDiscarded(false);
            double gaugeLoss = 0.0;
            int gaugeIterations = gaugeTrial < 2 ? 4 : 3;
            for (int iteration = 0;
                 iteration < gaugeIterations; ++iteration) {
                vector<int> previousValues = values;
                coordinatePass(iteration & 1, true);
                coordinatePass(!(iteration & 1), true);
                gaugeLoss = selectDiscarded(false);
                if (values == previousValues) break;
            }

            if (gaugeLoss < fittedLoss) {
                fittedLoss = gaugeLoss;
                bestValues = values;
                bestDiscarded = discarded;
            }
        }

        // Residual-priority basin: update vertices with the largest normalized
        // retained loss first, so inconsistent regions are repaired before
        // their neighbors and can induce a different top-D discard set.
        values = robustInitialValues;
        selectDiscarded(false);

        vector<double> vertexPriority(n, 0.0);
        for (int i = 0; i < m; ++i) {
            if (discarded[i]) continue;
            const Edge& e = edges[i];
            long long product = (long long)values[e.u] * values[e.v];
            double error = (double)e.w *
                           (double)llabs(product - (long long)e.x) /
                           (double)e.x;
            vertexPriority[e.u] += error;
            vertexPriority[e.v] += error;
        }

        vector<int> residualOrder(n);
        iota(residualOrder.begin(), residualOrder.end(), 0);
        sort(residualOrder.begin(), residualOrder.end(),
             [&](int a, int b) {
                 double scoreA = vertexPriority[a] /
                                 sqrt((double)max(1, degree[a]));
                 double scoreB = vertexPriority[b] /
                                 sqrt((double)max(1, degree[b]));
                 if (scoreA != scoreB) return scoreA > scoreB;
                 if (degree[a] != degree[b]) return degree[a] > degree[b];
                 return a < b;
             });

        double residualOrderLoss = 0.0;
        for (int iteration = 0; iteration < 6; ++iteration) {
            vector<int> previousValues = values;
            coordinatePass(iteration & 1, true, &residualOrder);
            residualOrderLoss = selectDiscarded(false);
            if (values == previousValues) break;
        }
        if (residualOrderLoss < fittedLoss) {
            fittedLoss = residualOrderLoss;
            bestValues = values;
            bestDiscarded = discarded;
        }

        // Deterministic perturbation restarts: modify a hashed subset of
        // coordinates, vary the Gauss-Seidel order, and polish each resulting
        // basin. Only strictly improving trimmed solutions are retained.
        if (m <= 250000) {
            vector<pair<uint64_t, int>> perturbKeys(n);
            vector<int> perturbOrder(n);

            // Build exact-product restart seeds from both sides of the trim
            // boundary. Marginal discarded edges may be clean observations,
            // while expensive retained edges offer the largest repair benefit.
            constexpr int DISCARDED_SEEDS = 10;
            constexpr int RETAINED_SEEDS = 10;
            priority_queue<pair<double, int>> marginalQueue;
            priority_queue<pair<double, int>,
                           vector<pair<double, int>>,
                           greater<pair<double, int>>> retainedQueue;
            for (int i = 0; i < m; ++i) {
                const Edge& e = edges[i];
                long long product =
                    (long long)bestValues[e.u] * bestValues[e.v];
                double error = (double)e.w *
                    (double)llabs(product - (long long)e.x) / (double)e.x;
                pair<double, int> candidate = {error, i};

                if (bestDiscarded[i]) {
                    if ((int)marginalQueue.size() < DISCARDED_SEEDS) {
                        marginalQueue.push(candidate);
                    } else if (candidate < marginalQueue.top()) {
                        marginalQueue.pop();
                        marginalQueue.push(candidate);
                    }
                } else {
                    if ((int)retainedQueue.size() < RETAINED_SEEDS) {
                        retainedQueue.push(candidate);
                    } else if (candidate > retainedQueue.top()) {
                        retainedQueue.pop();
                        retainedQueue.push(candidate);
                    }
                }
            }

            vector<int> marginalSeeds;
            while (!marginalQueue.empty()) {
                marginalSeeds.push_back(marginalQueue.top().second);
                marginalQueue.pop();
            }
            reverse(marginalSeeds.begin(), marginalSeeds.end());

            vector<int> retainedSeeds;
            while (!retainedQueue.empty()) {
                retainedSeeds.push_back(retainedQueue.top().second);
                retainedQueue.pop();
            }
            reverse(retainedSeeds.begin(), retainedSeeds.end());
            // Interleave the two seed classes. Optional search may terminate
            // early, so this ensures that both marginal discarded observations
            // and expensive retained observations receive restart attempts.
            vector<int> interleavedSeeds;
            interleavedSeeds.reserve(
                marginalSeeds.size() + retainedSeeds.size());
            size_t discardedPosition = 0;
            size_t retainedPosition = 0;
            while (discardedPosition < marginalSeeds.size() ||
                   retainedPosition < retainedSeeds.size()) {
                if (discardedPosition < marginalSeeds.size())
                    interleavedSeeds.push_back(
                        marginalSeeds[discardedPosition++]);
                if (retainedPosition < retainedSeeds.size())
                    interleavedSeeds.push_back(
                        retainedSeeds[retainedPosition++]);
            }
            marginalSeeds.swap(interleavedSeeds);

            // Use five diversified perturbations, then devote the remaining
            // budget to structurally meaningful exact factor-pair restarts.
            const int randomTrials = 5;
            const int totalTrials =
                randomTrials + (int)marginalSeeds.size();
            for (int trial = 0; trial < totalTrials; ++trial) {
                double elapsed =
                    (double)(clock() - searchStartTime) / CLOCKS_PER_SEC;
                if (elapsed > 8.95) break;

                values = bestValues;
                discarded = bestDiscarded;

                // Random trials use increasingly wide log-space moves. Later
                // trials are initialized exclusively by exact edge factors.
                double amplitude;
                if (trial < 2) amplitude = 0.10;
                else if (trial < 4) amplitude = 0.30;
                else amplitude = 0.70;

                for (int u = 0; u < n; ++u) {
                    uint64_t z =
                        (uint64_t)(u + 1) +
                        0x9e3779b97f4a7c15ULL * (uint64_t)(trial + 11);
                    z ^= z >> 30;
                    z *= 0xbf58476d1ce4e5b9ULL;
                    z ^= z >> 27;
                    z *= 0x94d049bb133111ebULL;
                    z ^= z >> 31;

                    perturbKeys[u] = {z, u};

                    // Only the first few trials use broad random moves; all
                    // remaining trials are reserved for exact-product seeds.
                    if (trial >= randomTrials || (z & 3ULL) >= 2ULL ||
                        degree[u] == 0)
                        continue;

                    double signedUnit =
                        ((double)((z >> 12) & 0xfffffULL) / 524287.5) - 1.0;
                    double factor = exp(amplitude * signedUnit);
                    long long candidate =
                        llround((double)values[u] * factor);
                    values[u] = (int)max(
                        1LL, min(1000000000LL, candidate));
                }

                // Snap a marginally discarded edge to the exact factor pair
                // nearest the current basin. This coordinated two-coordinate
                // move can survive trimming where either endpoint alone would
                // immediately be returned to the old coordinate optimum.
                if (trial >= randomTrials &&
                    trial - randomTrials < (int)marginalSeeds.size()) {
                    values = bestValues;
                    const Edge& seed =
                        edges[marginalSeeds[trial - randomTrials]];
                    int bestU = values[seed.u];
                    int bestV = values[seed.v];
                    double bestDistance =
                        numeric_limits<double>::infinity();

                    // Score each factorization on a compact neighborhood of
                    // strongly influential retained observations. This favors
                    // exact pairs that are compatible with local graph evidence
                    // instead of merely being close to the current coordinates.
                    constexpr int ANCHORS_PER_ENDPOINT = 12;
                    vector<int> anchorEdges;
                    anchorEdges.reserve(2 * ANCHORS_PER_ENDPOINT);

                    auto collectAnchors = [&](int endpoint) {
                        priority_queue<
                            pair<double, int>,
                            vector<pair<double, int>>,
                            greater<pair<double, int>>> strongest;

                        for (int p = offset[endpoint];
                             p < offset[endpoint + 1]; ++p) {
                            int edgeIndex = adjacency[p];
                            if (bestDiscarded[edgeIndex]) continue;

                            const Edge& e = edges[edgeIndex];
                            int other = e.u ^ e.v ^ endpoint;
                            double influence =
                                (double)e.w * bestValues[other] /
                                (double)e.x;
                            pair<double, int> entry =
                                {influence, edgeIndex};

                            if ((int)strongest.size() <
                                ANCHORS_PER_ENDPOINT) {
                                strongest.push(entry);
                            } else if (entry > strongest.top()) {
                                strongest.pop();
                                strongest.push(entry);
                            }
                        }

                        while (!strongest.empty()) {
                            anchorEdges.push_back(
                                strongest.top().second);
                            strongest.pop();
                        }
                    };

                    collectAnchors(seed.u);
                    collectAnchors(seed.v);
                    sort(anchorEdges.begin(), anchorEdges.end());
                    anchorEdges.erase(
                        unique(anchorEdges.begin(), anchorEdges.end()),
                        anchorEdges.end());

                    double bestAnchorLoss =
                        numeric_limits<double>::infinity();

                    auto considerPair = [&](int candidateU, int candidateV) {
                        double anchorLoss = 0.0;
                        for (int edgeIndex : anchorEdges) {
                            const Edge& e = edges[edgeIndex];
                            long long valueU =
                                e.u == seed.u ? candidateU :
                                e.u == seed.v ? candidateV :
                                bestValues[e.u];
                            long long valueV =
                                e.v == seed.u ? candidateU :
                                e.v == seed.v ? candidateV :
                                bestValues[e.v];
                            long long product = valueU * valueV;
                            anchorLoss +=
                                (double)e.w *
                                (double)llabs(
                                    product - (long long)e.x) /
                                (double)e.x;
                        }

                        double distance =
                            abs(log((double)candidateU /
                                    (double)bestValues[seed.u])) +
                            abs(log((double)candidateV /
                                    (double)bestValues[seed.v]));

                        if (anchorLoss < bestAnchorLoss - 1e-12 ||
                            (abs(anchorLoss - bestAnchorLoss) <= 1e-12 &&
                             distance < bestDistance)) {
                            bestAnchorLoss = anchorLoss;
                            bestDistance = distance;
                            bestU = candidateU;
                            bestV = candidateV;
                        }
                    };

                    for (int divisor = 1;
                         (long long)divisor * divisor <= seed.x;
                         ++divisor) {
                        if (seed.x % divisor != 0) continue;
                        int other = seed.x / divisor;
                        considerPair(divisor, other);
                        if (divisor != other)
                            considerPair(other, divisor);
                    }

                    values[seed.u] = bestU;
                    values[seed.v] = bestV;

                    // Propagate the coordinated exact seed through retained
                    // divisible observations. Only accept assignments close to
                    // the current basin, reducing cascades from a corrupted seed
                    // while allowing a clean edge to move an entire local region.
                    vector<char> recovered(n, 0);
                    vector<int> propagationQueue;
                    propagationQueue.reserve(n);
                    recovered[seed.u] = recovered[seed.v] = 1;
                    propagationQueue.push_back(seed.u);
                    propagationQueue.push_back(seed.v);

                    for (size_t head = 0;
                         head < propagationQueue.size(); ++head) {
                        int u = propagationQueue[head];
                        for (int p = offset[u]; p < offset[u + 1]; ++p) {
                            int edgeIndex = adjacency[p];
                            if (bestDiscarded[edgeIndex]) continue;

                            const Edge& e = edges[edgeIndex];
                            int v = e.u ^ e.v ^ u;
                            if (recovered[v]) continue;

                            vector<pair<int, int>> votes;
                            for (int q = offset[v]; q < offset[v + 1]; ++q) {
                                int otherEdgeIndex = adjacency[q];
                                if (bestDiscarded[otherEdgeIndex]) continue;
                                const Edge& otherEdge = edges[otherEdgeIndex];
                                int other =
                                    otherEdge.u ^ otherEdge.v ^ v;
                                if (!recovered[other] ||
                                    otherEdge.x % values[other] != 0)
                                    continue;
                                votes.push_back(
                                    {otherEdge.x / values[other],
                                     (int)otherEdge.w});
                            }
                            if (votes.empty()) continue;

                            sort(votes.begin(), votes.end());
                            int candidate = votes[0].first;
                            long long bestSupport = -1;
                            for (int left = 0;
                                 left < (int)votes.size();) {
                                int right = left;
                                long long support = 0;
                                while (right < (int)votes.size() &&
                                       votes[right].first ==
                                           votes[left].first) {
                                    support += votes[right].second;
                                    ++right;
                                }
                                if (support > bestSupport) {
                                    bestSupport = support;
                                    candidate = votes[left].first;
                                }
                                left = right;
                            }

                            double movement = abs(log(
                                (double)candidate /
                                (double)bestValues[v]));
                            // Exact divisibility votes are high-confidence and
                            // may need to cross a substantial gauge mismatch.
                            // Global loss evaluation still rejects bad cascades.
                            if (movement > 1.00) continue;

                            values[v] = candidate;
                            recovered[v] = 1;
                            propagationQueue.push_back(v);
                        }
                    }
                }

                sort(perturbKeys.begin(), perturbKeys.end());
                for (int i = 0; i < n; ++i)
                    perturbOrder[i] = perturbKeys[i].second;

                selectDiscarded(false);
                double perturbLoss = 0.0;
                for (int iteration = 0; iteration < 4; ++iteration) {
                    vector<int> previousValues = values;
                    coordinatePass(iteration & 1, true, &perturbOrder);
                    perturbLoss = selectDiscarded(false);
                    if (values == previousValues) break;
                }

                if (perturbLoss < fittedLoss) {
                    fittedLoss = perturbLoss;
                    bestValues = values;
                    bestDiscarded = discarded;
                }
            }
        }
    }

    // Polish the globally best basin to a stable trimmed coordinate optimum.
    // Fixed-set coordinate descent and subsequent top-D reselection are both
    // non-increasing operations for the trimmed objective.
    values = bestValues;
    discarded = bestDiscarded;
    double polishedLoss = fittedLoss;
    for (int iteration = 0; iteration < 6; ++iteration) {
        vector<int> previousValues = values;
        coordinatePass(iteration & 1, true);
        coordinatePass(!(iteration & 1), true);
        polishedLoss = selectDiscarded(false);
        if (values == previousValues) break;
    }
    if (polishedLoss + 1e-12 < fittedLoss) {
        fittedLoss = polishedLoss;
        bestValues = values;
        bestDiscarded = discarded;
    }

    // Recompute the selected solution directly over retained observations.
    // This avoids using a cancellation-affected incremental loss in the final
    // comparison against the safe all-ones candidate.
    values = bestValues;
    long double verifiedFittedLoss = 0.0L;
    for (int i = 0; i < m; ++i) {
        if (bestDiscarded[i]) continue;
        const Edge& e = edges[i];
        long long product =
            (long long)values[e.u] * values[e.v];
        verifiedFittedLoss +=
            (long double)e.w *
            (long double)llabs(product - (long long)e.x) /
            (long double)e.x;
    }
    fittedLoss = (double)verifiedFittedLoss;

    selectDiscarded(true);
    vector<char> baselineDiscarded = discarded;
    long double verifiedBaselineLoss = 0.0L;
    for (int i = 0; i < m; ++i) {
        if (baselineDiscarded[i]) continue;
        const Edge& e = edges[i];
        verifiedBaselineLoss +=
            (long double)e.w *
            (long double)(e.x - 1LL) /
            (long double)e.x;
    }

    if (verifiedBaselineLoss + 1e-12L < verifiedFittedLoss) {
        fill(values.begin(), values.end(), 1);
        discarded = baselineDiscarded;
    } else {
        discarded = bestDiscarded;
    }

    string output;
    output.reserve((size_t)n * 12 + (size_t)discardLimit * 9 + 64);
    for (int i = 0; i < n; ++i) {
        if (i) output.push_back(' ');
        output += to_string(values[i]);
    }
    output.push_back('\n');

    int outputDiscardCount = 0;
    for (char flag : discarded)
        outputDiscardCount += flag != 0;

    output += to_string(outputDiscardCount);
    for (int i = 0; i < m; ++i) {
        if (discarded[i]) {
            output.push_back(' ');
            output += to_string(i + 1);
        }
    }
    output.push_back('\n');
    fwrite(output.data(), 1, output.size(), stdout);
    return 0;
}
// EVOLVE-BLOCK-END
