static coroutine_fn int qcow2_co_preadv(BlockDriverState *bs, uint64_t offset,

                                        uint64_t bytes, QEMUIOVector *qiov,

                                        int flags)

{

    BDRVQcow2State *s = bs->opaque;

    int offset_in_cluster, n1;

    int ret;

    unsigned int cur_bytes; /* number of bytes in current iteration */

    uint64_t cluster_offset = 0;

    uint64_t bytes_done = 0;

    QEMUIOVector hd_qiov;

    uint8_t *cluster_data = NULL;



    qemu_iovec_init(&hd_qiov, qiov->niov);



    qemu_co_mutex_lock(&s->lock);



    while (bytes != 0) {



        /* prepare next request */

        cur_bytes = MIN(bytes, INT_MAX);

        if (s->crypto) {

            cur_bytes = MIN(cur_bytes,

                            QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size);

        }



        ret = qcow2_get_cluster_offset(bs, offset, &cur_bytes, &cluster_offset);

        if (ret < 0) {

            goto fail;

        }



        offset_in_cluster = offset_into_cluster(s, offset);



        qemu_iovec_reset(&hd_qiov);

        qemu_iovec_concat(&hd_qiov, qiov, bytes_done, cur_bytes);



        switch (ret) {

        case QCOW2_CLUSTER_UNALLOCATED:



            if (bs->backing) {

                /* read from the base image */

                n1 = qcow2_backing_read1(bs->backing->bs, &hd_qiov,

                                         offset, cur_bytes);

                if (n1 > 0) {

                    QEMUIOVector local_qiov;



                    qemu_iovec_init(&local_qiov, hd_qiov.niov);

                    qemu_iovec_concat(&local_qiov, &hd_qiov, 0, n1);



                    BLKDBG_EVENT(bs->file, BLKDBG_READ_BACKING_AIO);

                    qemu_co_mutex_unlock(&s->lock);

                    ret = bdrv_co_preadv(bs->backing, offset, n1,

                                         &local_qiov, 0);

                    qemu_co_mutex_lock(&s->lock);



                    qemu_iovec_destroy(&local_qiov);



                    if (ret < 0) {

                        goto fail;

                    }

                }

            } else {

                /* Note: in this case, no need to wait */

                qemu_iovec_memset(&hd_qiov, 0, 0, cur_bytes);

            }

            break;



        case QCOW2_CLUSTER_ZERO_PLAIN:

        case QCOW2_CLUSTER_ZERO_ALLOC:

            qemu_iovec_memset(&hd_qiov, 0, 0, cur_bytes);

            break;



        case QCOW2_CLUSTER_COMPRESSED:

            /* add AIO support for compressed blocks ? */

            ret = qcow2_decompress_cluster(bs, cluster_offset);

            if (ret < 0) {

                goto fail;

            }



            qemu_iovec_from_buf(&hd_qiov, 0,

                                s->cluster_cache + offset_in_cluster,

                                cur_bytes);

            break;



        case QCOW2_CLUSTER_NORMAL:

            if ((cluster_offset & 511) != 0) {

                ret = -EIO;

                goto fail;

            }



            if (bs->encrypted) {

                assert(s->crypto);



                /*

                 * For encrypted images, read everything into a temporary

                 * contiguous buffer on which the AES functions can work.

                 */

                if (!cluster_data) {

                    cluster_data =

                        qemu_try_blockalign(bs->file->bs,

                                            QCOW_MAX_CRYPT_CLUSTERS

                                            * s->cluster_size);

                    if (cluster_data == NULL) {

                        ret = -ENOMEM;

                        goto fail;

                    }

                }



                assert(cur_bytes <= QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size);

                qemu_iovec_reset(&hd_qiov);

                qemu_iovec_add(&hd_qiov, cluster_data, cur_bytes);

            }



            BLKDBG_EVENT(bs->file, BLKDBG_READ_AIO);

            qemu_co_mutex_unlock(&s->lock);

            ret = bdrv_co_preadv(bs->file,

                                 cluster_offset + offset_in_cluster,

                                 cur_bytes, &hd_qiov, 0);

            qemu_co_mutex_lock(&s->lock);

            if (ret < 0) {

                goto fail;

            }

            if (bs->encrypted) {

                assert(s->crypto);

                assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);

                assert((cur_bytes & (BDRV_SECTOR_SIZE - 1)) == 0);

                if (qcrypto_block_decrypt(s->crypto,

                                          (s->crypt_physical_offset ?

                                           cluster_offset + offset_in_cluster :

                                           offset),

                                          cluster_data,

                                          cur_bytes,

                                          NULL) < 0) {

                    ret = -EIO;

                    goto fail;

                }

                qemu_iovec_from_buf(qiov, bytes_done, cluster_data, cur_bytes);

            }

            break;



        default:

            g_assert_not_reached();

            ret = -EIO;

            goto fail;

        }



        bytes -= cur_bytes;

        offset += cur_bytes;

        bytes_done += cur_bytes;

    }

    ret = 0;



fail:

    qemu_co_mutex_unlock(&s->lock);



    qemu_iovec_destroy(&hd_qiov);

    qemu_vfree(cluster_data);



    return ret;

}
