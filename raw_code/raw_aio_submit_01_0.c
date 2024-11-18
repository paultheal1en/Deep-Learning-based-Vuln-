static BlockDriverAIOCB *raw_aio_submit(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int type)

{

    BDRVRawState *s = bs->opaque;



    if (fd_open(bs) < 0)

        return NULL;



    /*

     * If O_DIRECT is used the buffer needs to be aligned on a sector

     * boundary.  Check if this is the case or tell the low-level

     * driver that it needs to copy the buffer.

     */

    if ((bs->open_flags & BDRV_O_NOCACHE)) {

        if (!qiov_is_aligned(bs, qiov)) {

            type |= QEMU_AIO_MISALIGNED;

#ifdef CONFIG_LINUX_AIO

        } else if (s->use_aio) {

            return laio_submit(bs, s->aio_ctx, s->fd, sector_num, qiov,

                               nb_sectors, cb, opaque, type);

#endif

        }

    }



    return paio_submit(bs, s->fd, sector_num, qiov, nb_sectors,

                       cb, opaque, type);

}
