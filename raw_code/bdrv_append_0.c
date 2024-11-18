void bdrv_append(BlockDriverState *bs_new, BlockDriverState *bs_top,

                 Error **errp)

{

    Error *local_err = NULL;



    assert(!atomic_read(&bs_top->in_flight));

    assert(!atomic_read(&bs_new->in_flight));



    bdrv_set_backing_hd(bs_new, bs_top, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    change_parent_backing_link(bs_top, bs_new);



    /* bs_new is now referenced by its new parents, we don't need the

     * additional reference any more. */

out:

    bdrv_unref(bs_new);

}
