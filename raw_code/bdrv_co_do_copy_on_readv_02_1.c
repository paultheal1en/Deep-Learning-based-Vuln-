static int coroutine_fn bdrv_co_do_copy_on_readv(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, QEMUIOVector *qiov)

{

    /* Perform I/O through a temporary buffer so that users who scribble over

     * their read buffer while the operation is in progress do not end up

     * modifying the image file.  This is critical for zero-copy guest I/O

     * where anything might happen inside guest memory.

     */

    void *bounce_buffer;



    BlockDriver *drv = bs->drv;

    struct iovec iov;

    QEMUIOVector bounce_qiov;

    int64_t cluster_sector_num;

    int cluster_nb_sectors;

    size_t skip_bytes;

    int ret;



    /* Cover entire cluster so no additional backing file I/O is required when

     * allocating cluster in the image file.

     */

    bdrv_round_to_clusters(bs, sector_num, nb_sectors,

                           &cluster_sector_num, &cluster_nb_sectors);



    trace_bdrv_co_do_copy_on_readv(bs, sector_num, nb_sectors,

                                   cluster_sector_num, cluster_nb_sectors);



    iov.iov_len = cluster_nb_sectors * BDRV_SECTOR_SIZE;

    iov.iov_base = bounce_buffer = qemu_blockalign(bs, iov.iov_len);

    qemu_iovec_init_external(&bounce_qiov, &iov, 1);



    ret = drv->bdrv_co_readv(bs, cluster_sector_num, cluster_nb_sectors,

                             &bounce_qiov);

    if (ret < 0) {

        goto err;

    }



    if (drv->bdrv_co_write_zeroes &&

        buffer_is_zero(bounce_buffer, iov.iov_len)) {

        ret = bdrv_co_do_write_zeroes(bs, cluster_sector_num,

                                      cluster_nb_sectors, 0);

    } else {

        /* This does not change the data on the disk, it is not necessary

         * to flush even in cache=writethrough mode.

         */

        ret = drv->bdrv_co_writev(bs, cluster_sector_num, cluster_nb_sectors,

                                  &bounce_qiov);

    }



    if (ret < 0) {

        /* It might be okay to ignore write errors for guest requests.  If this

         * is a deliberate copy-on-read then we don't want to ignore the error.

         * Simply report it in all cases.

         */

        goto err;

    }



    skip_bytes = (sector_num - cluster_sector_num) * BDRV_SECTOR_SIZE;

    qemu_iovec_from_buf(qiov, 0, bounce_buffer + skip_bytes,

                        nb_sectors * BDRV_SECTOR_SIZE);



err:

    qemu_vfree(bounce_buffer);

    return ret;

}
