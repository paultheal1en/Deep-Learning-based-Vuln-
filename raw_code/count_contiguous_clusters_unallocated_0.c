static int count_contiguous_clusters_unallocated(int nb_clusters,

                                                 uint64_t *l2_table,

                                                 QCow2ClusterType wanted_type)

{

    int i;



    assert(wanted_type == QCOW2_CLUSTER_ZERO ||

           wanted_type == QCOW2_CLUSTER_UNALLOCATED);

    for (i = 0; i < nb_clusters; i++) {

        uint64_t entry = be64_to_cpu(l2_table[i]);

        QCow2ClusterType type = qcow2_get_cluster_type(entry);



        if (type != wanted_type || entry & L2E_OFFSET_MASK) {

            break;

        }

    }



    return i;

}
