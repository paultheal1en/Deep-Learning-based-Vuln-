static int count_contiguous_clusters_by_type(int nb_clusters,

                                             uint64_t *l2_table,

                                             int wanted_type)

{

    int i;



    for (i = 0; i < nb_clusters; i++) {

        int type = qcow2_get_cluster_type(be64_to_cpu(l2_table[i]));



        if (type != wanted_type) {

            break;

        }

    }



    return i;

}
