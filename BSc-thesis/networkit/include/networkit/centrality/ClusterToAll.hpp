/*
 * ClusterToAll.hpp
 *
 *  Created on: 21.04.2021
 *      Author: Florian Ernst
 */

#ifndef NETWORKIT_CENTRALITY_CLUSTER_TO_ALL_
#define NETWORKIT_CENTRALITY_CLUSTER_TO_ALL_

#include <networkit/centrality/Centrality.hpp>

namespace NetworKit {


/**
 * @ingroup centrality
 * Approximation of the Edge centralities vie clustering
 */
class ClusterToAll final: public Centrality {

public:

    ClusterToAll(const Graph &G, std::vector<node> cluster_centers);

    /**
     * Computes betweenness approximation on the graph passed in constructor.
     */
    void run() override;


private:

    bool computeEdgeCentrality = true;
    std::vector<node> cluster_centers;
    count r; // number of samples
    count cluster_length; // number of relevant clusters
};

} /* namespace NetworKit */

#endif // NETWORKIT_CENTRALITY_CLUSTER_TO_ALL_
