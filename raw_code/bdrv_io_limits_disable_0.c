void bdrv_io_limits_disable(BlockDriverState *bs)

{

    bs->io_limits_enabled = false;



    bdrv_start_throttled_reqs(bs);



    throttle_destroy(&bs->throttle_state);

}
