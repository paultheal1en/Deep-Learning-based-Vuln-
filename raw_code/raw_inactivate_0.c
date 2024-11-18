static int raw_inactivate(BlockDriverState *bs)

{

    int ret;

    uint64_t perm = 0;

    uint64_t shared = BLK_PERM_ALL;



    ret = raw_handle_perm_lock(bs, RAW_PL_PREPARE, perm, shared, NULL);

    if (ret) {

        return ret;

    }

    raw_handle_perm_lock(bs, RAW_PL_COMMIT, perm, shared, NULL);

    return 0;

}
