static void bdrv_qed_drain(BlockDriverState *bs)

{

    BDRVQEDState *s = bs->opaque;



    /* Cancel timer and start doing I/O that were meant to happen as if it

     * fired, that way we get bdrv_drain() taking care of the ongoing requests

     * correctly. */

    qed_cancel_need_check_timer(s);

    qed_plug_allocating_write_reqs(s);

    bdrv_aio_flush(s->bs, qed_clear_need_check, s);

}
