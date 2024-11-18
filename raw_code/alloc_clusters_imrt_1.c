static int64_t alloc_clusters_imrt(BlockDriverState *bs,

                                   int cluster_count,

                                   uint16_t **refcount_table,

                                   int64_t *imrt_nb_clusters,

                                   int64_t *first_free_cluster)

{

    BDRVQcowState *s = bs->opaque;

    int64_t cluster = *first_free_cluster, i;

    bool first_gap = true;

    int contiguous_free_clusters;

    int ret;



    /* Starting at *first_free_cluster, find a range of at least cluster_count

     * continuously free clusters */

    for (contiguous_free_clusters = 0;

         cluster < *imrt_nb_clusters &&

         contiguous_free_clusters < cluster_count;

         cluster++)

    {

        if (!(*refcount_table)[cluster]) {

            contiguous_free_clusters++;

            if (first_gap) {

                /* If this is the first free cluster found, update

                 * *first_free_cluster accordingly */

                *first_free_cluster = cluster;

                first_gap = false;

            }

        } else if (contiguous_free_clusters) {

            contiguous_free_clusters = 0;

        }

    }



    /* If contiguous_free_clusters is greater than zero, it contains the number

     * of continuously free clusters until the current cluster; the first free

     * cluster in the current "gap" is therefore

     * cluster - contiguous_free_clusters */



    /* If no such range could be found, grow the in-memory refcount table

     * accordingly to append free clusters at the end of the image */

    if (contiguous_free_clusters < cluster_count) {

        /* contiguous_free_clusters clusters are already empty at the image end;

         * we need cluster_count clusters; therefore, we have to allocate

         * cluster_count - contiguous_free_clusters new clusters at the end of

         * the image (which is the current value of cluster; note that cluster

         * may exceed old_imrt_nb_clusters if *first_free_cluster pointed beyond

         * the image end) */

        ret = realloc_refcount_array(s, refcount_table, imrt_nb_clusters,

                                     cluster + cluster_count

                                     - contiguous_free_clusters);

        if (ret < 0) {

            return ret;

        }

    }



    /* Go back to the first free cluster */

    cluster -= contiguous_free_clusters;

    for (i = 0; i < cluster_count; i++) {

        (*refcount_table)[cluster + i] = 1;

    }



    return cluster << s->cluster_bits;

}
