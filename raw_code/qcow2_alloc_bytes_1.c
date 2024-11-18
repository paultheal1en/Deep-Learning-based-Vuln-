int64_t qcow2_alloc_bytes(BlockDriverState *bs, int size)

{

    BDRVQcowState *s = bs->opaque;

    int64_t offset;

    size_t free_in_cluster;

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_CLUSTER_ALLOC_BYTES);

    assert(size > 0 && size <= s->cluster_size);

    assert(!s->free_byte_offset || offset_into_cluster(s, s->free_byte_offset));



    offset = s->free_byte_offset;



    if (offset) {

        uint64_t refcount;

        ret = qcow2_get_refcount(bs, offset >> s->cluster_bits, &refcount);

        if (ret < 0) {

            return ret;

        }



        if (refcount == s->refcount_max) {

            offset = 0;

        }

    }



    free_in_cluster = s->cluster_size - offset_into_cluster(s, offset);

    if (!offset || free_in_cluster < size) {

        int64_t new_cluster = alloc_clusters_noref(bs, s->cluster_size);

        if (new_cluster < 0) {

            return new_cluster;

        }



        if (!offset || ROUND_UP(offset, s->cluster_size) != new_cluster) {

            offset = new_cluster;

        }

    }



    assert(offset);

    ret = update_refcount(bs, offset, size, 1, false, QCOW2_DISCARD_NEVER);

    if (ret < 0) {

        return ret;

    }



    /* The cluster refcount was incremented; refcount blocks must be flushed

     * before the caller's L2 table updates. */

    qcow2_cache_set_dependency(bs, s->l2_table_cache, s->refcount_block_cache);



    s->free_byte_offset = offset + size;

    if (!offset_into_cluster(s, s->free_byte_offset)) {

        s->free_byte_offset = 0;

    }



    return offset;

}
