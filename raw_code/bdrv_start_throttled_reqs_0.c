static bool bdrv_start_throttled_reqs(BlockDriverState *bs)

{

    bool drained = false;

    bool enabled = bs->io_limits_enabled;

    int i;



    bs->io_limits_enabled = false;



    for (i = 0; i < 2; i++) {

        while (qemu_co_enter_next(&bs->throttled_reqs[i])) {

            drained = true;

        }

    }



    bs->io_limits_enabled = enabled;



    return drained;

}
