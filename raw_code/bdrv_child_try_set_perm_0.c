int bdrv_child_try_set_perm(BdrvChild *c, uint64_t perm, uint64_t shared,

                            Error **errp)

{

    int ret;



    ret = bdrv_child_check_perm(c, perm, shared, errp);

    if (ret < 0) {

        bdrv_child_abort_perm_update(c);

        return ret;

    }



    bdrv_child_set_perm(c, perm, shared);



    return 0;

}
