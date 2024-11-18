static coroutine_fn int qcow2_co_writev(BlockDriverState *bs,

                           int64_t sector_num,

                           int remaining_sectors,

                           QEMUIOVector *qiov)

{

    BDRVQcowState *s = bs->opaque;

    int index_in_cluster;

    int n_end;

    int ret;

    int cur_nr_sectors; /* number of sectors in current iteration */

    uint64_t cluster_offset;

    QEMUIOVector hd_qiov;

    uint64_t bytes_done = 0;

    uint8_t *cluster_data = NULL;

    QCowL2Meta *l2meta;



    trace_qcow2_writev_start_req(qemu_coroutine_self(), sector_num,

                                 remaining_sectors);



    qemu_iovec_init(&hd_qiov, qiov->niov);



    s->cluster_cache_offset = -1; /* disable compressed cache */



    qemu_co_mutex_lock(&s->lock);



    while (remaining_sectors != 0) {



        l2meta = NULL;



        trace_qcow2_writev_start_part(qemu_coroutine_self());

        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        n_end = index_in_cluster + remaining_sectors;

        if (s->crypt_method &&

            n_end > QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors) {

            n_end = QCOW_MAX_CRYPT_CLUSTERS * s->cluster_sectors;

        }



        ret = qcow2_alloc_cluster_offset(bs, sector_num << 9,

            index_in_cluster, n_end, &cur_nr_sectors, &cluster_offset, &l2meta);

        if (ret < 0) {

            goto fail;

        }



        if (l2meta->nb_clusters > 0 &&

            (s->compatible_features & QCOW2_COMPAT_LAZY_REFCOUNTS)) {

            qcow2_mark_dirty(bs);

        }



        assert((cluster_offset & 511) == 0);



        qemu_iovec_reset(&hd_qiov);

        qemu_iovec_concat(&hd_qiov, qiov, bytes_done,

            cur_nr_sectors * 512);



        if (s->crypt_method) {

            if (!cluster_data) {

                cluster_data = qemu_blockalign(bs, QCOW_MAX_CRYPT_CLUSTERS *

                                                 s->cluster_size);

            }



            assert(hd_qiov.size <=

                   QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size);

            qemu_iovec_to_buf(&hd_qiov, 0, cluster_data, hd_qiov.size);



            qcow2_encrypt_sectors(s, sector_num, cluster_data,

                cluster_data, cur_nr_sectors, 1, &s->aes_encrypt_key);



            qemu_iovec_reset(&hd_qiov);

            qemu_iovec_add(&hd_qiov, cluster_data,

                cur_nr_sectors * 512);

        }



        qemu_co_mutex_unlock(&s->lock);

        BLKDBG_EVENT(bs->file, BLKDBG_WRITE_AIO);

        trace_qcow2_writev_data(qemu_coroutine_self(),

                                (cluster_offset >> 9) + index_in_cluster);

        ret = bdrv_co_writev(bs->file,

                             (cluster_offset >> 9) + index_in_cluster,

                             cur_nr_sectors, &hd_qiov);

        qemu_co_mutex_lock(&s->lock);

        if (ret < 0) {

            goto fail;

        }



        if (l2meta != NULL) {

            ret = qcow2_alloc_cluster_link_l2(bs, l2meta);

            if (ret < 0) {

                goto fail;

            }



            run_dependent_requests(s, l2meta);

            g_free(l2meta);

            l2meta = NULL;

        }



        remaining_sectors -= cur_nr_sectors;

        sector_num += cur_nr_sectors;

        bytes_done += cur_nr_sectors * 512;

        trace_qcow2_writev_done_part(qemu_coroutine_self(), cur_nr_sectors);

    }

    ret = 0;



fail:

    if (l2meta != NULL) {

        run_dependent_requests(s, l2meta);

        g_free(l2meta);

    }



    qemu_co_mutex_unlock(&s->lock);



    qemu_iovec_destroy(&hd_qiov);

    qemu_vfree(cluster_data);

    trace_qcow2_writev_done_req(qemu_coroutine_self(), ret);



    return ret;

}
