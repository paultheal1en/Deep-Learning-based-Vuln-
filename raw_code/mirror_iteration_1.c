static int coroutine_fn mirror_iteration(MirrorBlockJob *s)

{

    BlockDriverState *source = s->common.bs;

    BlockDriverState *target = s->target;

    QEMUIOVector qiov;

    int ret, nb_sectors;

    int64_t end;

    struct iovec iov;



    end = s->common.len >> BDRV_SECTOR_BITS;

    s->sector_num = bdrv_get_next_dirty(source, s->sector_num);

    nb_sectors = MIN(BDRV_SECTORS_PER_DIRTY_CHUNK, end - s->sector_num);

    bdrv_reset_dirty(source, s->sector_num, nb_sectors);



    /* Copy the dirty cluster.  */

    iov.iov_base = s->buf;

    iov.iov_len  = nb_sectors * 512;

    qemu_iovec_init_external(&qiov, &iov, 1);



    trace_mirror_one_iteration(s, s->sector_num, nb_sectors);

    ret = bdrv_co_readv(source, s->sector_num, nb_sectors, &qiov);

    if (ret < 0) {

        return ret;

    }

    return bdrv_co_writev(target, s->sector_num, nb_sectors, &qiov);

}
