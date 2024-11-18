void bdrv_io_limits_enable(BlockDriverState *bs, const char *group)

{

    assert(!bs->throttle_state);

    throttle_group_register_bs(bs, group);

}
