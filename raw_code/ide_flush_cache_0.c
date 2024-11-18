void ide_flush_cache(IDEState *s)

{

    if (s->bs == NULL) {

        ide_flush_cb(s, 0);

        return;

    }



    s->status |= BUSY_STAT;

    block_acct_start(bdrv_get_stats(s->bs), &s->acct, 0, BLOCK_ACCT_FLUSH);

    s->pio_aiocb = bdrv_aio_flush(s->bs, ide_flush_cb, s);

}
