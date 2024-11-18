static coroutine_fn int qcow_co_readv(BlockDriverState *bs, int64_t sector_num,

                         int nb_sectors, QEMUIOVector *qiov)

{

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster;

    int ret = 0, n;

    uint64_t cluster_offset;

    struct iovec hd_iov;

    QEMUIOVector hd_qiov;

    uint8_t *buf;

    void *orig_buf;



    if (qiov->niov > 1) {

        buf = orig_buf = qemu_try_blockalign(bs, qiov->size);

        if (buf == NULL) {

            return -ENOMEM;

        }

    } else {

        orig_buf = NULL;

        buf = (uint8_t *)qiov->iov->iov_base;

    }



    qemu_co_mutex_lock(&s->lock);



    while (nb_sectors != 0) {

        /* prepare next request */

        cluster_offset = get_cluster_offset(bs, sector_num << 9,

                                                 0, 0, 0, 0);

        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        n = s->cluster_sectors - index_in_cluster;

        if (n > nb_sectors) {

            n = nb_sectors;

        }



        if (!cluster_offset) {

            if (bs->backing) {

                /* read from the base image */

                hd_iov.iov_base = (void *)buf;

                hd_iov.iov_len = n * 512;

                qemu_iovec_init_external(&hd_qiov, &hd_iov, 1);

                qemu_co_mutex_unlock(&s->lock);

                ret = bdrv_co_readv(bs->backing, sector_num, n, &hd_qiov);

                qemu_co_mutex_lock(&s->lock);

                if (ret < 0) {

                    goto fail;

                }

            } else {

                /* Note: in this case, no need to wait */

                memset(buf, 0, 512 * n);

            }

        } else if (cluster_offset & QCOW_OFLAG_COMPRESSED) {

            /* add AIO support for compressed blocks ? */

            if (decompress_cluster(bs, cluster_offset) < 0) {

                goto fail;

            }

            memcpy(buf,

                   s->cluster_cache + index_in_cluster * 512, 512 * n);

        } else {

            if ((cluster_offset & 511) != 0) {

                goto fail;

            }

            hd_iov.iov_base = (void *)buf;

            hd_iov.iov_len = n * 512;

            qemu_iovec_init_external(&hd_qiov, &hd_iov, 1);

            qemu_co_mutex_unlock(&s->lock);

            ret = bdrv_co_readv(bs->file,

                                (cluster_offset >> 9) + index_in_cluster,

                                n, &hd_qiov);

            qemu_co_mutex_lock(&s->lock);

            if (ret < 0) {

                break;

            }

            if (bs->encrypted) {

                assert(s->crypto);

                if (qcrypto_block_decrypt(s->crypto, sector_num, buf,

                                          n * BDRV_SECTOR_SIZE, NULL) < 0) {

                    goto fail;

                }

            }

        }

        ret = 0;



        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;

    }



done:

    qemu_co_mutex_unlock(&s->lock);



    if (qiov->niov > 1) {

        qemu_iovec_from_buf(qiov, 0, orig_buf, qiov->size);

        qemu_vfree(orig_buf);

    }



    return ret;



fail:

    ret = -EIO;

    goto done;

}
