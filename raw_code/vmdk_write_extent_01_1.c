static int vmdk_write_extent(VmdkExtent *extent, int64_t cluster_offset,

                            int64_t offset_in_cluster, QEMUIOVector *qiov,

                            uint64_t qiov_offset, uint64_t n_bytes,

                            uint64_t offset)

{

    int ret;

    VmdkGrainMarker *data = NULL;

    uLongf buf_len;

    QEMUIOVector local_qiov;

    struct iovec iov;

    int64_t write_offset;

    int64_t write_end_sector;



    if (extent->compressed) {

        void *compressed_data;



        if (!extent->has_marker) {

            ret = -EINVAL;

            goto out;

        }

        buf_len = (extent->cluster_sectors << 9) * 2;

        data = g_malloc(buf_len + sizeof(VmdkGrainMarker));



        compressed_data = g_malloc(n_bytes);

        qemu_iovec_to_buf(qiov, qiov_offset, compressed_data, n_bytes);

        ret = compress(data->data, &buf_len, compressed_data, n_bytes);

        g_free(compressed_data);



        if (ret != Z_OK || buf_len == 0) {

            ret = -EINVAL;

            goto out;

        }



        data->lba = offset >> BDRV_SECTOR_BITS;

        data->size = buf_len;



        n_bytes = buf_len + sizeof(VmdkGrainMarker);

        iov = (struct iovec) {

            .iov_base   = data,

            .iov_len    = n_bytes,

        };

        qemu_iovec_init_external(&local_qiov, &iov, 1);

    } else {

        qemu_iovec_init(&local_qiov, qiov->niov);

        qemu_iovec_concat(&local_qiov, qiov, qiov_offset, n_bytes);

    }



    write_offset = cluster_offset + offset_in_cluster,

    ret = bdrv_co_pwritev(extent->file, write_offset, n_bytes,

                          &local_qiov, 0);



    write_end_sector = DIV_ROUND_UP(write_offset + n_bytes, BDRV_SECTOR_SIZE);



    if (extent->compressed) {

        extent->next_cluster_sector = write_end_sector;

    } else {

        extent->next_cluster_sector = MAX(extent->next_cluster_sector,

                                          write_end_sector);

    }



    if (ret < 0) {

        goto out;

    }

    ret = 0;

 out:

    g_free(data);

    if (!extent->compressed) {

        qemu_iovec_destroy(&local_qiov);

    }

    return ret;

}
