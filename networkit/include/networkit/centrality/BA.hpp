/*
 * BA.hpp
 *
 *  Created on: 21.04.2021
 *      Author: Florian Ernst
 */

#ifndef NETWORKIT_CENTRALITY_B_A_
#define NETWORKIT_CENTRALITY_B_A_

#include <networkit/centrality/Centrality.hpp>

namespace NetworKit {


/**
 * @ingroup centrality
 * Approximation of the Edge centralities vie clustering
 */
class BA final: public Centrality {

public:

    BA(const Graph &G, std::vector<node> cluster_centers);

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

#endif // NETWORKIT_CENTRALITY_B_A_
