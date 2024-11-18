static inline void flash_sync_area(Flash *s, int64_t off, int64_t len)

{

    int64_t start, end, nb_sectors;

    QEMUIOVector iov;



    if (!s->bdrv || bdrv_is_read_only(s->bdrv)) {

        return;

    }



    assert(!(len % BDRV_SECTOR_SIZE));

    start = off / BDRV_SECTOR_SIZE;

    end = (off + len) / BDRV_SECTOR_SIZE;

    nb_sectors = end - start;

    qemu_iovec_init(&iov, 1);

    qemu_iovec_add(&iov, s->storage + (start * BDRV_SECTOR_SIZE),

                                        nb_sectors * BDRV_SECTOR_SIZE);

    bdrv_aio_writev(s->bdrv, start, &iov, nb_sectors, bdrv_sync_complete, NULL);

}
