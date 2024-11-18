static void flash_sync_page(Flash *s, int page)

{

    QEMUIOVector *iov = g_new(QEMUIOVector, 1);



    if (!s->blk || blk_is_read_only(s->blk)) {

        return;

    }



    qemu_iovec_init(iov, 1);

    qemu_iovec_add(iov, s->storage + page * s->pi->page_size,

                   s->pi->page_size);

    blk_aio_pwritev(s->blk, page * s->pi->page_size, iov, 0,

                    blk_sync_complete, iov);

}
