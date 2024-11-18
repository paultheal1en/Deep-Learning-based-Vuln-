static coroutine_fn int qcow_co_writev(BlockDriverState *bs, int64_t sector_num,

                          int nb_sectors, QEMUIOVector *qiov)

{

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster;

    uint64_t cluster_offset;

    int ret = 0, n;

    struct iovec hd_iov;

    QEMUIOVector hd_qiov;

    uint8_t *buf;

    void *orig_buf;



    s->cluster_cache_offset = -1; /* disable compressed cache */



    /* We must always copy the iov when encrypting, so we

     * don't modify the original data buffer during encryption */

    if (bs->encrypted || qiov->niov > 1) {

        buf = orig_buf = qemu_try_blockalign(bs, qiov->size);

        if (buf == NULL) {

            return -ENOMEM;

        }

        qemu_iovec_to_buf(qiov, 0, buf, qiov->size);

    } else {

        orig_buf = NULL;

        buf = (uint8_t *)qiov->iov->iov_base;

    }



    qemu_co_mutex_lock(&s->lock);



    while (nb_sectors != 0) {



        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        n = s->cluster_sectors - index_in_cluster;

        if (n > nb_sectors) {

            n = nb_sectors;

        }

        cluster_offset = get_cluster_offset(bs, sector_num << 9, 1, 0,

                                            index_in_cluster,

                                            index_in_cluster + n);

        if (!cluster_offset || (cluster_offset & 511) != 0) {

            ret = -EIO;

            break;

        }

        if (bs->encrypted) {

            assert(s->crypto);

            if (qcrypto_block_encrypt(s->crypto, sector_num, buf,

                                      n * BDRV_SECTOR_SIZE, NULL) < 0) {

                ret = -EIO;

                break;

            }

        }



        hd_iov.iov_base = (void *)buf;

        hd_iov.iov_len = n * 512;

        qemu_iovec_init_external(&hd_qiov, &hd_iov, 1);

        qemu_co_mutex_unlock(&s->lock);

        ret = bdrv_co_writev(bs->file,

                             (cluster_offset >> 9) + index_in_cluster,

                             n, &hd_qiov);

        qemu_co_mutex_lock(&s->lock);

        if (ret < 0) {

            break;

        }

        ret = 0;



        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;

    }

    qemu_co_mutex_unlock(&s->lock);



    qemu_vfree(orig_buf);



    return ret;

}
