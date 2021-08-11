/*
 * ClusterToAll.hpp
 *
 *  Created on: 21.04.2021
 *      Author: Florian Ernst
 */

#ifndef NETWORKIT_CENTRALITY_CLUSTER_TO_ALL2_
#define NETWORKIT_CENTRALITY_CLUSTER_TO_ALL2_

#include <networkit/centrality/Centrality.hpp>

namespace NetworKit {


/**
 * @ingroup centrality
 * Approximation of the Edge centralities vie clustering
 */
class ClusterToAll2 final: public Centrality {

public:

    ClusterToAll2(const Graph &G, std::vector<node> cluster_centers, std::vector<float> node_weights);

    /**
     * Computes betweenness approximation on the graph passed in constructor.
     */
    void run() override;


private:

    bool computeEdgeCentrality = true;
    std::vector<node> cluster_centers;
    std::vector<float> node_weights;
    count r; // number of samples
    count cluster_length; // number of relevant clusters
};

} /* namespace NetworKit */

#endif // NETWORKIT_CENTRALITY_CLUSTER_TO_ALL2_
