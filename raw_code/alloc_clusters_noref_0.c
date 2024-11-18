static int64_t alloc_clusters_noref(BlockDriverState *bs, int64_t size)

{

    BDRVQcowState *s = bs->opaque;

    int i, nb_clusters;



    nb_clusters = size_to_clusters(s, size);

retry:

    for(i = 0; i < nb_clusters; i++) {

        int64_t i = s->free_cluster_index++;

        if (get_refcount(bs, i) != 0)

            goto retry;

    }

#ifdef DEBUG_ALLOC2

    printf("alloc_clusters: size=%" PRId64 " -> %" PRId64 "\n",

            size,

            (s->free_cluster_index - nb_clusters) << s->cluster_bits);

#endif

    return (s->free_cluster_index - nb_clusters) << s->cluster_bits;

}
