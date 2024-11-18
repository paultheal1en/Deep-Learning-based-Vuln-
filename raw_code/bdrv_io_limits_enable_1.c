void bdrv_io_limits_enable(BlockDriverState *bs)

{

    assert(!bs->io_limits_enabled);

    throttle_init(&bs->throttle_state,

                  bdrv_get_aio_context(bs),

                  QEMU_CLOCK_VIRTUAL,

                  bdrv_throttle_read_timer_cb,

                  bdrv_throttle_write_timer_cb,

                  bs);

    bs->io_limits_enabled = true;

}
