static int coroutine_fn do_perform_cow(BlockDriverState *bs,

                                       uint64_t src_cluster_offset,

                                       uint64_t cluster_offset,

                                       int offset_in_cluster,

                                       int bytes)

{

    BDRVQcow2State *s = bs->opaque;

    QEMUIOVector qiov;

    struct iovec iov;

    int ret;



    iov.iov_len = bytes;

    iov.iov_base = qemu_try_blockalign(bs, iov.iov_len);

    if (iov.iov_base == NULL) {

        return -ENOMEM;

    }



    qemu_iovec_init_external(&qiov, &iov, 1);



    BLKDBG_EVENT(bs->file, BLKDBG_COW_READ);



    if (!bs->drv) {

        ret = -ENOMEDIUM;

        goto out;

    }



    /* Call .bdrv_co_readv() directly instead of using the public block-layer

     * interface.  This avoids double I/O throttling and request tracking,

     * which can lead to deadlock when block layer copy-on-read is enabled.

     */

    ret = bs->drv->bdrv_co_preadv(bs, src_cluster_offset + offset_in_cluster,

                                  bytes, &qiov, 0);

    if (ret < 0) {

        goto out;

    }



    if (bs->encrypted) {

        Error *err = NULL;

        int64_t sector = (cluster_offset + offset_in_cluster)

                         >> BDRV_SECTOR_BITS;

        assert(s->cipher);

        assert((offset_in_cluster & ~BDRV_SECTOR_MASK) == 0);

        assert((bytes & ~BDRV_SECTOR_MASK) == 0);

        if (qcow2_encrypt_sectors(s, sector, iov.iov_base, iov.iov_base,

                                  bytes >> BDRV_SECTOR_BITS, true, &err) < 0) {

            ret = -EIO;

            error_free(err);

            goto out;

        }

    }



    ret = qcow2_pre_write_overlap_check(bs, 0,

            cluster_offset + offset_in_cluster, bytes);

    if (ret < 0) {

        goto out;

    }



    BLKDBG_EVENT(bs->file, BLKDBG_COW_WRITE);

    ret = bdrv_co_pwritev(bs->file->bs, cluster_offset + offset_in_cluster,

                          bytes, &qiov, 0);

    if (ret < 0) {

        goto out;

    }



    ret = 0;

out:

    qemu_vfree(iov.iov_base);

    return ret;

}
