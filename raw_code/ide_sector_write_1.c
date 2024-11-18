void ide_sector_write(IDEState *s)
{
    int64_t sector_num;
    int n;
    s->status = READY_STAT | SEEK_STAT | BUSY_STAT;
    sector_num = ide_get_sector(s);
#if defined(DEBUG_IDE)
    printf("sector=%" PRId64 "\n", sector_num);
#endif
    n = s->nsector;
    if (n > s->req_nb_sectors) {
        n = s->req_nb_sectors;
    s->iov.iov_base = s->io_buffer;
    s->iov.iov_len  = n * BDRV_SECTOR_SIZE;
    qemu_iovec_init_external(&s->qiov, &s->iov, 1);
    bdrv_acct_start(s->bs, &s->acct, n * BDRV_SECTOR_SIZE, BDRV_ACCT_READ);
    s->pio_aiocb = bdrv_aio_writev(s->bs, sector_num, &s->qiov, n,
                                   ide_sector_write_cb, s);