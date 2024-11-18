static coroutine_fn int sd_co_writev(BlockDriverState *bs, int64_t sector_num,

                        int nb_sectors, QEMUIOVector *qiov)

{

    SheepdogAIOCB acb;

    int ret;

    int64_t offset = (sector_num + nb_sectors) * BDRV_SECTOR_SIZE;

    BDRVSheepdogState *s = bs->opaque;



    if (offset > s->inode.vdi_size) {

        ret = sd_truncate(bs, offset);

        if (ret < 0) {

            return ret;

        }

    }



    sd_aio_setup(&acb, s, qiov, sector_num, nb_sectors, AIOCB_WRITE_UDATA);



retry:

    if (check_overlapping_aiocb(s, &acb)) {

        qemu_co_queue_wait(&s->overlapping_queue);

        goto retry;

    }



    sd_co_rw_vector(&acb);

    sd_write_done(&acb);



    QLIST_REMOVE(&acb, aiocb_siblings);

    qemu_co_queue_restart_all(&s->overlapping_queue);

    return acb.ret;

}
