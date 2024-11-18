static int qcow2_create2(const char *filename, int64_t total_size,

                         const char *backing_file, const char *backing_format,

                         int flags, size_t cluster_size, int prealloc,

                         QEMUOptionParameter *options, int version,

                         Error **errp)

{

    /* Calculate cluster_bits */

    int cluster_bits;

    cluster_bits = ffs(cluster_size) - 1;

    if (cluster_bits < MIN_CLUSTER_BITS || cluster_bits > MAX_CLUSTER_BITS ||

        (1 << cluster_bits) != cluster_size)

    {

        error_setg(errp, "Cluster size must be a power of two between %d and "

                   "%dk", 1 << MIN_CLUSTER_BITS, 1 << (MAX_CLUSTER_BITS - 10));

        return -EINVAL;

    }



    /*

     * Open the image file and write a minimal qcow2 header.

     *

     * We keep things simple and start with a zero-sized image. We also

     * do without refcount blocks or a L1 table for now. We'll fix the

     * inconsistency later.

     *

     * We do need a refcount table because growing the refcount table means

     * allocating two new refcount blocks - the seconds of which would be at

     * 2 GB for 64k clusters, and we don't want to have a 2 GB initial file

     * size for any qcow2 image.

     */

    BlockDriverState* bs;

    QCowHeader *header;

    uint8_t* refcount_table;

    Error *local_err = NULL;

    int ret;



    ret = bdrv_create_file(filename, options, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        return ret;

    }



    bs = NULL;

    ret = bdrv_open(&bs, filename, NULL, NULL, BDRV_O_RDWR | BDRV_O_PROTOCOL,

                    NULL, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        return ret;

    }



    /* Write the header */

    QEMU_BUILD_BUG_ON((1 << MIN_CLUSTER_BITS) < sizeof(*header));

    header = g_malloc0(cluster_size);

    *header = (QCowHeader) {

        .magic                      = cpu_to_be32(QCOW_MAGIC),

        .version                    = cpu_to_be32(version),

        .cluster_bits               = cpu_to_be32(cluster_bits),

        .size                       = cpu_to_be64(0),

        .l1_table_offset            = cpu_to_be64(0),

        .l1_size                    = cpu_to_be32(0),

        .refcount_table_offset      = cpu_to_be64(cluster_size),

        .refcount_table_clusters    = cpu_to_be32(1),

        .refcount_order             = cpu_to_be32(3 + REFCOUNT_SHIFT),

        .header_length              = cpu_to_be32(sizeof(*header)),

    };



    if (flags & BLOCK_FLAG_ENCRYPT) {

        header->crypt_method = cpu_to_be32(QCOW_CRYPT_AES);

    } else {

        header->crypt_method = cpu_to_be32(QCOW_CRYPT_NONE);

    }



    if (flags & BLOCK_FLAG_LAZY_REFCOUNTS) {

        header->compatible_features |=

            cpu_to_be64(QCOW2_COMPAT_LAZY_REFCOUNTS);

    }



    ret = bdrv_pwrite(bs, 0, header, cluster_size);

    g_free(header);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write qcow2 header");

        goto out;

    }



    /* Write an empty refcount table */

    refcount_table = g_malloc0(cluster_size);

    ret = bdrv_pwrite(bs, cluster_size, refcount_table, cluster_size);

    g_free(refcount_table);



    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not write refcount table");

        goto out;

    }



    bdrv_unref(bs);

    bs = NULL;



    /*

     * And now open the image and make it consistent first (i.e. increase the

     * refcount of the cluster that is occupied by the header and the refcount

     * table)

     */

    BlockDriver* drv = bdrv_find_format("qcow2");

    assert(drv != NULL);

    ret = bdrv_open(&bs, filename, NULL, NULL,

        BDRV_O_RDWR | BDRV_O_CACHE_WB | BDRV_O_NO_FLUSH, drv, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto out;

    }



    ret = qcow2_alloc_clusters(bs, 2 * cluster_size);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not allocate clusters for qcow2 "

                         "header and refcount table");

        goto out;



    } else if (ret != 0) {

        error_report("Huh, first cluster in empty image is already in use?");

        abort();

    }



    /* Okay, now that we have a valid image, let's give it the right size */

    ret = bdrv_truncate(bs, total_size * BDRV_SECTOR_SIZE);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not resize image");

        goto out;

    }



    /* Want a backing file? There you go.*/

    if (backing_file) {

        ret = bdrv_change_backing_file(bs, backing_file, backing_format);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Could not assign backing file '%s' "

                             "with format '%s'", backing_file, backing_format);

            goto out;

        }

    }



    /* And if we're supposed to preallocate metadata, do that now */

    if (prealloc) {

        BDRVQcowState *s = bs->opaque;

        qemu_co_mutex_lock(&s->lock);

        ret = preallocate(bs);

        qemu_co_mutex_unlock(&s->lock);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Could not preallocate metadata");

            goto out;

        }

    }



    bdrv_unref(bs);

    bs = NULL;



    /* Reopen the image without BDRV_O_NO_FLUSH to flush it before returning */

    ret = bdrv_open(&bs, filename, NULL, NULL,

                    BDRV_O_RDWR | BDRV_O_CACHE_WB | BDRV_O_NO_BACKING,

                    drv, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    ret = 0;

out:

    if (bs) {

        bdrv_unref(bs);

    }

    return ret;

}
