static int bdrv_inactivate_recurse(BlockDriverState *bs,

                                   bool setting_flag)

{

    BdrvChild *child, *parent;

    int ret;



    if (!setting_flag && bs->drv->bdrv_inactivate) {

        ret = bs->drv->bdrv_inactivate(bs);

        if (ret < 0) {

            return ret;

        }

    }



    if (setting_flag) {

        uint64_t perm, shared_perm;



        bs->open_flags |= BDRV_O_INACTIVE;



        QLIST_FOREACH(parent, &bs->parents, next_parent) {

            if (parent->role->inactivate) {

                ret = parent->role->inactivate(parent);

                if (ret < 0) {

                    bs->open_flags &= ~BDRV_O_INACTIVE;

                    return ret;

                }

            }

        }



        /* Update permissions, they may differ for inactive nodes */

        bdrv_get_cumulative_perm(bs, &perm, &shared_perm);

        bdrv_check_perm(bs, perm, shared_perm, NULL, &error_abort);

        bdrv_set_perm(bs, perm, shared_perm);

    }



    QLIST_FOREACH(child, &bs->children, next) {

        ret = bdrv_inactivate_recurse(child->bs, setting_flag);

        if (ret < 0) {

            return ret;

        }

    }



    /* At this point persistent bitmaps should be already stored by the format

     * driver */

    bdrv_release_persistent_dirty_bitmaps(bs);



    return 0;

}
