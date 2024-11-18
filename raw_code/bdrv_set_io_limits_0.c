void bdrv_set_io_limits(BlockDriverState *bs,

                        ThrottleConfig *cfg)

{

    int i;



    throttle_config(&bs->throttle_state, cfg);



    for (i = 0; i < 2; i++) {

        qemu_co_enter_next(&bs->throttled_reqs[i]);

    }

}
