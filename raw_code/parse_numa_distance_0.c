static void parse_numa_distance(NumaDistOptions *dist, Error **errp)

{

    uint16_t src = dist->src;

    uint16_t dst = dist->dst;

    uint8_t val = dist->val;



    if (src >= MAX_NODES || dst >= MAX_NODES) {

        error_setg(errp,

                   "Invalid node %" PRIu16

                   ", max possible could be %" PRIu16,

                   MAX(src, dst), MAX_NODES);

        return;

    }



    if (!numa_info[src].present || !numa_info[dst].present) {

        error_setg(errp, "Source/Destination NUMA node is missing. "

                   "Please use '-numa node' option to declare it first.");

        return;

    }



    if (val < NUMA_DISTANCE_MIN) {

        error_setg(errp, "NUMA distance (%" PRIu8 ") is invalid, "

                   "it shouldn't be less than %d.",

                   val, NUMA_DISTANCE_MIN);

        return;

    }



    if (src == dst && val != NUMA_DISTANCE_MIN) {

        error_setg(errp, "Local distance of node %d should be %d.",

                   src, NUMA_DISTANCE_MIN);

        return;

    }



    numa_info[src].distance[dst] = val;

    have_numa_distance = true;

}
