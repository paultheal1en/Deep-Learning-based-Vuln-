static int64_t alloc_refcount_block(BlockDriverState *bs, int64_t cluster_index)

{

    BDRVQcowState *s = bs->opaque;

    unsigned int refcount_table_index;

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC);



    /* Find the refcount block for the given cluster */

    refcount_table_index = cluster_index >> (s->cluster_bits - REFCOUNT_SHIFT);



    if (refcount_table_index < s->refcount_table_size) {



        uint64_t refcount_block_offset =

            s->refcount_table[refcount_table_index];



        /* If it's already there, we're done */

        if (refcount_block_offset) {

            if (refcount_block_offset != s->refcount_block_cache_offset) {

                ret = load_refcount_block(bs, refcount_block_offset);

                if (ret < 0) {

                    return ret;

                }

            }

            return refcount_block_offset;

        }

    }



    /*

     * If we came here, we need to allocate something. Something is at least

     * a cluster for the new refcount block. It may also include a new refcount

     * table if the old refcount table is too small.

     *

     * Note that allocating clusters here needs some special care:

     *

     * - We can't use the normal qcow2_alloc_clusters(), it would try to

     *   increase the refcount and very likely we would end up with an endless

     *   recursion. Instead we must place the refcount blocks in a way that

     *   they can describe them themselves.

     *

     * - We need to consider that at this point we are inside update_refcounts

     *   and doing the initial refcount increase. This means that some clusters

     *   have already been allocated by the caller, but their refcount isn't

     *   accurate yet. free_cluster_index tells us where this allocation ends

     *   as long as we don't overwrite it by freeing clusters.

     *

     * - alloc_clusters_noref and qcow2_free_clusters may load a different

     *   refcount block into the cache

     */



    if (cache_refcount_updates) {

        ret = write_refcount_block(bs);

        if (ret < 0) {

            return ret;

        }

    }



    /* Allocate the refcount block itself and mark it as used */

    uint64_t new_block = alloc_clusters_noref(bs, s->cluster_size);

    memset(s->refcount_block_cache, 0, s->cluster_size);

    s->refcount_block_cache_offset = new_block;



#ifdef DEBUG_ALLOC2

    fprintf(stderr, "qcow2: Allocate refcount block %d for %" PRIx64

        " at %" PRIx64 "\n",

        refcount_table_index, cluster_index << s->cluster_bits, new_block);

#endif



    if (in_same_refcount_block(s, new_block, cluster_index << s->cluster_bits)) {

        /* The block describes itself, need to update the cache */

        int block_index = (new_block >> s->cluster_bits) &

            ((1 << (s->cluster_bits - REFCOUNT_SHIFT)) - 1);

        s->refcount_block_cache[block_index] = cpu_to_be16(1);

    } else {

        /* Described somewhere else. This can recurse at most twice before we

         * arrive at a block that describes itself. */

        ret = update_refcount(bs, new_block, s->cluster_size, 1);

        if (ret < 0) {

            goto fail_block;

        }

    }



    /* Now the new refcount block needs to be written to disk */

    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_WRITE);

    ret = bdrv_pwrite(bs->file, new_block, s->refcount_block_cache,

        s->cluster_size);

    if (ret < 0) {

        goto fail_block;

    }



    /* If the refcount table is big enough, just hook the block up there */

    if (refcount_table_index < s->refcount_table_size) {

        uint64_t data64 = cpu_to_be64(new_block);

        BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_HOOKUP);

        ret = bdrv_pwrite(bs->file,

            s->refcount_table_offset + refcount_table_index * sizeof(uint64_t),

            &data64, sizeof(data64));

        if (ret < 0) {

            goto fail_block;

        }



        s->refcount_table[refcount_table_index] = new_block;

        return new_block;

    }



    /*

     * If we come here, we need to grow the refcount table. Again, a new

     * refcount table needs some space and we can't simply allocate to avoid

     * endless recursion.

     *

     * Therefore let's grab new refcount blocks at the end of the image, which

     * will describe themselves and the new refcount table. This way we can

     * reference them only in the new table and do the switch to the new

     * refcount table at once without producing an inconsistent state in

     * between.

     */

    BLKDBG_EVENT(bs->file, BLKDBG_REFTABLE_GROW);



    /* Calculate the number of refcount blocks needed so far */

    uint64_t refcount_block_clusters = 1 << (s->cluster_bits - REFCOUNT_SHIFT);

    uint64_t blocks_used = (s->free_cluster_index +

        refcount_block_clusters - 1) / refcount_block_clusters;



    /* And now we need at least one block more for the new metadata */

    uint64_t table_size = next_refcount_table_size(s, blocks_used + 1);

    uint64_t last_table_size;

    uint64_t blocks_clusters;

    do {

        uint64_t table_clusters = size_to_clusters(s, table_size);

        blocks_clusters = 1 +

            ((table_clusters + refcount_block_clusters - 1)

            / refcount_block_clusters);

        uint64_t meta_clusters = table_clusters + blocks_clusters;



        last_table_size = table_size;

        table_size = next_refcount_table_size(s, blocks_used +

            ((meta_clusters + refcount_block_clusters - 1)

            / refcount_block_clusters));



    } while (last_table_size != table_size);



#ifdef DEBUG_ALLOC2

    fprintf(stderr, "qcow2: Grow refcount table %" PRId32 " => %" PRId64 "\n",

        s->refcount_table_size, table_size);

#endif



    /* Create the new refcount table and blocks */

    uint64_t meta_offset = (blocks_used * refcount_block_clusters) *

        s->cluster_size;

    uint64_t table_offset = meta_offset + blocks_clusters * s->cluster_size;

    uint16_t *new_blocks = qemu_mallocz(blocks_clusters * s->cluster_size);

    uint64_t *new_table = qemu_mallocz(table_size * sizeof(uint64_t));



    assert(meta_offset >= (s->free_cluster_index * s->cluster_size));



    /* Fill the new refcount table */

    memcpy(new_table, s->refcount_table,

        s->refcount_table_size * sizeof(uint64_t));

    new_table[refcount_table_index] = new_block;



    int i;

    for (i = 0; i < blocks_clusters; i++) {

        new_table[blocks_used + i] = meta_offset + (i * s->cluster_size);

    }



    /* Fill the refcount blocks */

    uint64_t table_clusters = size_to_clusters(s, table_size * sizeof(uint64_t));

    int block = 0;

    for (i = 0; i < table_clusters + blocks_clusters; i++) {

        new_blocks[block++] = cpu_to_be16(1);

    }



    /* Write refcount blocks to disk */

    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_WRITE_BLOCKS);

    ret = bdrv_pwrite(bs->file, meta_offset, new_blocks,

        blocks_clusters * s->cluster_size);

    qemu_free(new_blocks);

    if (ret < 0) {

        goto fail_table;

    }



    /* Write refcount table to disk */

    for(i = 0; i < table_size; i++) {

        cpu_to_be64s(&new_table[i]);

    }



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_WRITE_TABLE);

    ret = bdrv_pwrite(bs->file, table_offset, new_table,

        table_size * sizeof(uint64_t));

    if (ret < 0) {

        goto fail_table;

    }



    for(i = 0; i < table_size; i++) {

        cpu_to_be64s(&new_table[i]);

    }



    /* Hook up the new refcount table in the qcow2 header */

    uint8_t data[12];

    cpu_to_be64w((uint64_t*)data, table_offset);

    cpu_to_be32w((uint32_t*)(data + 8), table_clusters);

    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC_SWITCH_TABLE);

    ret = bdrv_pwrite(bs->file, offsetof(QCowHeader, refcount_table_offset),

        data, sizeof(data));

    if (ret < 0) {

        goto fail_table;

    }



    /* And switch it in memory */

    uint64_t old_table_offset = s->refcount_table_offset;

    uint64_t old_table_size = s->refcount_table_size;



    qemu_free(s->refcount_table);

    s->refcount_table = new_table;

    s->refcount_table_size = table_size;

    s->refcount_table_offset = table_offset;



    /* Free old table. Remember, we must not change free_cluster_index */

    uint64_t old_free_cluster_index = s->free_cluster_index;

    qcow2_free_clusters(bs, old_table_offset, old_table_size * sizeof(uint64_t));

    s->free_cluster_index = old_free_cluster_index;



    ret = load_refcount_block(bs, new_block);

    if (ret < 0) {

        goto fail_block;

    }



    return new_block;



fail_table:

    qemu_free(new_table);

fail_block:

    s->refcount_block_cache_offset = 0;

    return ret;

}
