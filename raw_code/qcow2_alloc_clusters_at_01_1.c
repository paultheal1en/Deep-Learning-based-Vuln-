int qcow2_alloc_clusters_at(BlockDriverState *bs, uint64_t offset,
    int nb_clusters)
{
    BDRVQcowState *s = bs->opaque;
    uint64_t cluster_index;
    uint64_t old_free_cluster_index;
    int i, refcount, ret;
    /* Check how many clusters there are free */
    cluster_index = offset >> s->cluster_bits;
    for(i = 0; i < nb_clusters; i++) {
        refcount = get_refcount(bs, cluster_index++);
        if (refcount < 0) {
            return refcount;
        } else if (refcount != 0) {
            break;
        }
    }
    /* And then allocate them */
    ret = update_refcount(bs, offset, i << s->cluster_bits, 1);
    if (ret < 0) {
        return ret;
    }
    return i;
}