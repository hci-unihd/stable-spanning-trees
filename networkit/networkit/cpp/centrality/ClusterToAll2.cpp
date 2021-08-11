/*
 * ClusterToAll2.cpp
 *
 *  Created on: 21.04.2021
 *      Author: Florian Ernst
 */

#include <networkit/centrality/ClusterToAll2.hpp>
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

ClusterToAll2::ClusterToAll2(const Graph& G, std::vector<node> cluster_centers, std::vector<float> node_weights) : Centrality(G, false, true), cluster_centers(cluster_centers), node_weights(node_weights) {

}


void ClusterToAll2::run() {
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


    cluster_length = cluster_centers.size();

    handler.assureRunning();
    #pragma omp parallel
    {
        auto ssspPtr = G.isWeighted() ? std::unique_ptr<SSSP>(new Dijkstra(G, 0, true, false))
                                      : std::unique_ptr<SSSP>(new BFS(G, 0, true, false));

#pragma omp for
        for (omp_index i = 1; i <= static_cast<omp_index>(cluster_length); i++) {
            // sample paths between the cluster centers
            node u = cluster_centers[i-1];


            auto &sssp = *ssspPtr;
            sssp.setSource(u);
            if (!handler.isRunning()) continue;
            sssp.run();
            if (!handler.isRunning()) continue;
            // iterate over all possible target nodes
            for (node v = 0; v < G.upperNodeIdBound(); v++)
            {
                if (!G.hasNode(v)) continue;

                if (u == v) continue;

                node t = v;

                if (!handler.isRunning()) continue;
                if (sssp.numberOfPaths(v) > 0)
                {
                    // at least one path between {u, v} exists
                    std::queue<node> q;
                    for (node pred : sssp.getPredecessors(t))
                    {
                        q.push(pred);
                        const edgeid edgeId = G.edgeId(pred, t);
                        
                        float increment = std::max(node_weights[t], node_weights[pred]);
                        
#pragma omp atomic
                        edgeScoreData[edgeId] += increment;

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
                            if (pred != u)
#pragma omp atomic
                                scoreData[pred] += 1.;
                                
                            const edgeid edgeId = G.edgeId(pred, vertex);
                            
                            float increment = std::max(node_weights[t], node_weights[vertex]);
                            
 #pragma omp atomic
                            edgeScoreData[edgeId] += increment;

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
