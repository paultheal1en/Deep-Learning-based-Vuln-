static void raw_invalidate_cache(BlockDriverState *bs, Error **errp)

{

    BDRVRawState *s = bs->opaque;

    int ret;



    assert(!(bdrv_get_flags(bs) & BDRV_O_INACTIVE));

    ret = raw_handle_perm_lock(bs, RAW_PL_PREPARE, s->perm, s->shared_perm,

                               errp);

    if (ret) {

        return;

    }

    raw_handle_perm_lock(bs, RAW_PL_COMMIT, s->perm, s->shared_perm, NULL);

}
