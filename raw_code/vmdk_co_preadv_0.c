vmdk_co_preadv(BlockDriverState *bs, uint64_t offset, uint64_t bytes,

               QEMUIOVector *qiov, int flags)

{

    BDRVVmdkState *s = bs->opaque;

    int ret;

    uint64_t n_bytes, offset_in_cluster;

    VmdkExtent *extent = NULL;

    QEMUIOVector local_qiov;

    uint64_t cluster_offset;

    uint64_t bytes_done = 0;



    qemu_iovec_init(&local_qiov, qiov->niov);

    qemu_co_mutex_lock(&s->lock);



    while (bytes > 0) {

        extent = find_extent(s, offset >> BDRV_SECTOR_BITS, extent);

        if (!extent) {

            ret = -EIO;

            goto fail;

        }

        ret = get_cluster_offset(bs, extent, NULL,

                                 offset, false, &cluster_offset, 0, 0);

        offset_in_cluster = vmdk_find_offset_in_cluster(extent, offset);



        n_bytes = MIN(bytes, extent->cluster_sectors * BDRV_SECTOR_SIZE

                             - offset_in_cluster);



        if (ret != VMDK_OK) {

            /* if not allocated, try to read from parent image, if exist */

            if (bs->backing && ret != VMDK_ZEROED) {

                if (!vmdk_is_cid_valid(bs)) {

                    ret = -EINVAL;

                    goto fail;

                }



                qemu_iovec_reset(&local_qiov);

                qemu_iovec_concat(&local_qiov, qiov, bytes_done, n_bytes);



                ret = bdrv_co_preadv(bs->backing->bs, offset, n_bytes,

                                     &local_qiov, 0);

                if (ret < 0) {

                    goto fail;

                }

            } else {

                qemu_iovec_memset(qiov, bytes_done, 0, n_bytes);

            }

        } else {

            qemu_iovec_reset(&local_qiov);

            qemu_iovec_concat(&local_qiov, qiov, bytes_done, n_bytes);



            ret = vmdk_read_extent(extent, cluster_offset, offset_in_cluster,

                                   &local_qiov, n_bytes);

            if (ret) {

                goto fail;

            }

        }

        bytes -= n_bytes;

        offset += n_bytes;

        bytes_done += n_bytes;

    }



    ret = 0;

fail:

    qemu_co_mutex_unlock(&s->lock);

    qemu_iovec_destroy(&local_qiov);



    return ret;

}
