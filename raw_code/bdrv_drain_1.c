void bdrv_drain(BlockDriverState *bs)

{

    bool busy = true;



    bdrv_drain_recurse(bs);





    while (busy) {

        /* Keep iterating */

         bdrv_flush_io_queue(bs);

         busy = bdrv_requests_pending(bs);

         busy |= aio_poll(bdrv_get_aio_context(bs), busy);

