static int count_cow_clusters(BDRVQcow2State *s, int nb_clusters,

    uint64_t *l2_table, int l2_index)

{

    int i;



    for (i = 0; i < nb_clusters; i++) {

        uint64_t l2_entry = be64_to_cpu(l2_table[l2_index + i]);

        QCow2ClusterType cluster_type = qcow2_get_cluster_type(l2_entry);



        switch(cluster_type) {

        case QCOW2_CLUSTER_NORMAL:

            if (l2_entry & QCOW_OFLAG_COPIED) {

                goto out;

            }

            break;

        case QCOW2_CLUSTER_UNALLOCATED:

        case QCOW2_CLUSTER_COMPRESSED:

        case QCOW2_CLUSTER_ZERO:

            break;

        default:

            abort();

        }

    }



out:

    assert(i <= nb_clusters);

    return i;

}
