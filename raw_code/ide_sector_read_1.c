void ide_sector_read(IDEState *s)
{
    int64_t sector_num;
    int n;
    s->status = READY_STAT | SEEK_STAT;
    s->error = 0; /* not needed by IDE spec, but needed by Windows */
    sector_num = ide_get_sector(s);
    n = s->nsector;
    if (n == 0) {
        ide_transfer_stop(s);
    s->status |= BUSY_STAT;
    if (n > s->req_nb_sectors) {
        n = s->req_nb_sectors;
#if defined(DEBUG_IDE)
    printf("sector=%" PRId64 "\n", sector_num);
#endif
    s->iov.iov_base = s->io_buffer;
    s->iov.iov_len  = n * BDRV_SECTOR_SIZE;
    qemu_iovec_init_external(&s->qiov, &s->iov, 1);
    bdrv_acct_start(s->bs, &s->acct, n * BDRV_SECTOR_SIZE, BDRV_ACCT_READ);
    s->pio_aiocb = bdrv_aio_readv(s->bs, sector_num, &s->qiov, n,
                                  ide_sector_read_cb, s);