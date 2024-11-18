static int make_completely_empty(BlockDriverState *bs)

{

    BDRVQcow2State *s = bs->opaque;

    int ret, l1_clusters;

    int64_t offset;

    uint64_t *new_reftable = NULL;

    uint64_t rt_entry, l1_size2;

    struct {

        uint64_t l1_offset;

        uint64_t reftable_offset;

        uint32_t reftable_clusters;

    } QEMU_PACKED l1_ofs_rt_ofs_cls;



    ret = qcow2_cache_empty(bs, s->l2_table_cache);

    if (ret < 0) {

        goto fail;

    }



    ret = qcow2_cache_empty(bs, s->refcount_block_cache);

    if (ret < 0) {

        goto fail;

    }



    /* Refcounts will be broken utterly */

    ret = qcow2_mark_dirty(bs);

    if (ret < 0) {

        goto fail;

    }



    BLKDBG_EVENT(bs->file, BLKDBG_L1_UPDATE);



    l1_clusters = DIV_ROUND_UP(s->l1_size, s->cluster_size / sizeof(uint64_t));

    l1_size2 = (uint64_t)s->l1_size * sizeof(uint64_t);



    /* After this call, neither the in-memory nor the on-disk refcount

     * information accurately describe the actual references */



    ret = bdrv_pwrite_zeroes(bs->file, s->l1_table_offset,

                             l1_clusters * s->cluster_size, 0);

    if (ret < 0) {

        goto fail_broken_refcounts;

    }

    memset(s->l1_table, 0, l1_size2);



    BLKDBG_EVENT(bs->file, BLKDBG_EMPTY_IMAGE_PREPARE);



    /* Overwrite enough clusters at the beginning of the sectors to place

     * the refcount table, a refcount block and the L1 table in; this may

     * overwrite parts of the existing refcount and L1 table, which is not

     * an issue because the dirty flag is set, complete data loss is in fact

     * desired and partial data loss is consequently fine as well */

    ret = bdrv_pwrite_zeroes(bs->file, s->cluster_size,

                             (2 + l1_clusters) * s->cluster_size, 0);

    /* This call (even if it failed overall) may have overwritten on-disk

     * refcount structures; in that case, the in-memory refcount information

     * will probably differ from the on-disk information which makes the BDS

     * unusable */

    if (ret < 0) {

        goto fail_broken_refcounts;

    }



    BLKDBG_EVENT(bs->file, BLKDBG_L1_UPDATE);

    BLKDBG_EVENT(bs->file, BLKDBG_REFTABLE_UPDATE);



    /* "Create" an empty reftable (one cluster) directly after the image

     * header and an empty L1 table three clusters after the image header;

     * the cluster between those two will be used as the first refblock */

    l1_ofs_rt_ofs_cls.l1_offset = cpu_to_be64(3 * s->cluster_size);

    l1_ofs_rt_ofs_cls.reftable_offset = cpu_to_be64(s->cluster_size);

    l1_ofs_rt_ofs_cls.reftable_clusters = cpu_to_be32(1);

    ret = bdrv_pwrite_sync(bs->file, offsetof(QCowHeader, l1_table_offset),

                           &l1_ofs_rt_ofs_cls, sizeof(l1_ofs_rt_ofs_cls));

    if (ret < 0) {

        goto fail_broken_refcounts;

    }



    s->l1_table_offset = 3 * s->cluster_size;



    new_reftable = g_try_new0(uint64_t, s->cluster_size / sizeof(uint64_t));

    if (!new_reftable) {

        ret = -ENOMEM;

        goto fail_broken_refcounts;

    }



    s->refcount_table_offset = s->cluster_size;

    s->refcount_table_size   = s->cluster_size / sizeof(uint64_t);




    g_free(s->refcount_table);

    s->refcount_table = new_reftable;

    new_reftable = NULL;



    /* Now the in-memory refcount information again corresponds to the on-disk

     * information (reftable is empty and no refblocks (the refblock cache is

     * empty)); however, this means some clusters (e.g. the image header) are

     * referenced, but not refcounted, but the normal qcow2 code assumes that

     * the in-memory information is always correct */



    BLKDBG_EVENT(bs->file, BLKDBG_REFBLOCK_ALLOC);



    /* Enter the first refblock into the reftable */

    rt_entry = cpu_to_be64(2 * s->cluster_size);

    ret = bdrv_pwrite_sync(bs->file, s->cluster_size,

                           &rt_entry, sizeof(rt_entry));

    if (ret < 0) {

        goto fail_broken_refcounts;

    }

    s->refcount_table[0] = 2 * s->cluster_size;



    s->free_cluster_index = 0;

    assert(3 + l1_clusters <= s->refcount_block_size);

    offset = qcow2_alloc_clusters(bs, 3 * s->cluster_size + l1_size2);

    if (offset < 0) {

        ret = offset;

        goto fail_broken_refcounts;

    } else if (offset > 0) {

        error_report("First cluster in emptied image is in use");

        abort();

    }



    /* Now finally the in-memory information corresponds to the on-disk

     * structures and is correct */

    ret = qcow2_mark_clean(bs);

    if (ret < 0) {

        goto fail;

    }



    ret = bdrv_truncate(bs->file->bs, (3 + l1_clusters) * s->cluster_size);

    if (ret < 0) {

        goto fail;

    }



    return 0;



fail_broken_refcounts:

    /* The BDS is unusable at this point. If we wanted to make it usable, we

     * would have to call qcow2_refcount_close(), qcow2_refcount_init(),

     * qcow2_check_refcounts(), qcow2_refcount_close() and qcow2_refcount_init()

     * again. However, because the functions which could have caused this error

     * path to be taken are used by those functions as well, it's very likely

     * that that sequence will fail as well. Therefore, just eject the BDS. */

    bs->drv = NULL;



fail:

    g_free(new_reftable);

    return ret;

}