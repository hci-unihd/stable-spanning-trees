/*
 * BA.cpp
 *
 *  Created on: 21.04.2021
 *      Author: Florian Ernst
 */

#include <networkit/centrality/BA.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/distance/Diameter.hpp>
#include <networkit/graph/GraphTools.hpp>
#include <networkit/distance/Dijkstra.hpp>
#include <networkit/distance/BFS.hpp>
#include <networkit/distance/SSSP.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <omp.h>

namespace NetworKit {

BA::BA(const Graph& G, std::vector<node> cluster_centers) : Centrality(G, false, true), cluster_centers(cluster_centers) {

}


void BA::run() {
    Aux::SignalHandler handler;
    scoreData.clear();
    scoreData.resize(G.upperNodeIdBound());

    count z2 = G.upperEdgeIdBound();
    edgeScoreData.clear();
    edgeScoreData.resize(z2);

    // Initialize the edgeScoreData array
        
        for(int i = 0; i<z2; i++) {
        	edgeScoreData[i] = 0;
        }

    // TODO: Ende nach r/2 möglich...

    r = cluster_centers.size() * cluster_centers.size();
    cluster_length = cluster_centers.size();

    handler.assureRunning();
    #pragma omp parallel
    {
        auto ssspPtr = G.isWeighted() ? std::unique_ptr<SSSP>(new Dijkstra(G, 0, true, false))
                                      : std::unique_ptr<SSSP>(new BFS(G, 0, true, false));

#pragma omp for
        for (omp_index i = 1; i <= static_cast<omp_index>(r); i++) {
            // sample paths between the cluster centers
            node u, v;
            u = cluster_centers[(i-1) / cluster_length];
            v = cluster_centers[(i-1) % cluster_length];
            if (u == v)
            {
                continue;
            }

            auto &sssp = *ssspPtr;
            sssp.setSource(u);
            sssp.setTarget(v);

            if (!handler.isRunning()) continue;
            sssp.run();
            if (!handler.isRunning()) continue;
            if (sssp.numberOfPaths(v) > 0) { // at least one path between {u, v} exists
                // random path sampling and estimation update
                node t = v;

                // I can use the usual algorithm
                // TODO could be impoved
                if (sssp.numberOfPaths(v) == 1)
                {
                    while (t != u)  
                    {
                        // sample z in P_u(t) with probability sigma_uz / sigma_us
                        std::vector<std::pair<node, double> > choices;
                        for (node z : sssp.getPredecessors(t)) {
                            bigfloat tmp = sssp.numberOfPaths(z) / sssp.numberOfPaths(t);
                            double weight;
                            tmp.ToDouble(weight);
                            choices.emplace_back(z, weight); 	// sigma_uz / sigma_us
                        }
                        node z = Aux::Random::weightedChoice(choices);
                        assert (z <= G.upperNodeIdBound());

                        const edgeid edgeId = G.edgeId(z, t);
#pragma omp atomic
                        edgeScoreData[edgeId] += 1.;


                        if (z != u)
#pragma omp atomic
                            scoreData[z] += 1.;

                        t = z;
                    }
                }
                
                else
                {
                    std::queue<node> q;
                    for (node pred : sssp.getPredecessors(t))
                    {
                        q.push(pred);
                        const edgeid edgeId = G.edgeId(pred, t);
#pragma omp atomic
                        edgeScoreData[edgeId] += 1.;

                        if (pred != u)
#pragma omp atomic
                            scoreData[pred] += 1.;
                    }
                    while (q.size() != 0)
                    {
                        node vertex = q.front();
                        q.pop();
                        for (node pred : sssp.getPredecessors(vertex))
                        {
                            q.push(pred);
                            const edgeid edgeId = G.edgeId(pred, vertex);
#pragma omp atomic
                            edgeScoreData[edgeId] += 1.;

                            if (pred != u)
#pragma omp atomic
                                scoreData[pred] += 1.;
                        }
                    }
                    

                }

            }
        }
    }
    handler.assureRunning();

    hasRun = true;
}


} /* namespace NetworKit */
