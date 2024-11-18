void qed_acquire(BDRVQEDState *s)

{

    aio_context_acquire(bdrv_get_aio_context(s->bs));

}
