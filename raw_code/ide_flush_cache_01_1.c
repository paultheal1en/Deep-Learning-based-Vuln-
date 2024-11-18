void ide_flush_cache(IDEState *s)

{

    if (s->bs == NULL) {

        ide_flush_cb(s, 0);

        return;

    }




    bdrv_acct_start(s->bs, &s->acct, 0, BDRV_ACCT_FLUSH);

    bdrv_aio_flush(s->bs, ide_flush_cb, s);

}