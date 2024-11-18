static bool bdrv_drain_one(BlockDriverState *bs)

{

    bool bs_busy;



    bdrv_flush_io_queue(bs);

    bdrv_start_throttled_reqs(bs);

    bs_busy = bdrv_requests_pending(bs);

    bs_busy |= aio_poll(bdrv_get_aio_context(bs), bs_busy);

    return bs_busy;

}
