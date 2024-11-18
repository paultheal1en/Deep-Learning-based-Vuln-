void bdrv_swap(BlockDriverState *bs_new, BlockDriverState *bs_old)

{

    BlockDriverState tmp;



    /* The code needs to swap the node_name but simply swapping node_list won't

     * work so first remove the nodes from the graph list, do the swap then

     * insert them back if needed.

     */

    if (bs_new->node_name[0] != '\0') {

        QTAILQ_REMOVE(&graph_bdrv_states, bs_new, node_list);

    }

    if (bs_old->node_name[0] != '\0') {

        QTAILQ_REMOVE(&graph_bdrv_states, bs_old, node_list);

    }



    /* bs_new must be anonymous and shouldn't have anything fancy enabled */

    assert(bs_new->device_name[0] == '\0');

    assert(QLIST_EMPTY(&bs_new->dirty_bitmaps));

    assert(bs_new->job == NULL);

    assert(bs_new->dev == NULL);

    assert(bdrv_op_blocker_is_empty(bs_new));

    assert(bs_new->io_limits_enabled == false);

    assert(!throttle_have_timer(&bs_new->throttle_state));



    tmp = *bs_new;

    *bs_new = *bs_old;

    *bs_old = tmp;



    /* there are some fields that should not be swapped, move them back */

    bdrv_move_feature_fields(&tmp, bs_old);

    bdrv_move_feature_fields(bs_old, bs_new);

    bdrv_move_feature_fields(bs_new, &tmp);



    /* bs_new shouldn't be in bdrv_states even after the swap!  */

    assert(bs_new->device_name[0] == '\0');



    /* Check a few fields that should remain attached to the device */

    assert(bs_new->dev == NULL);

    assert(bs_new->job == NULL);

    assert(bdrv_op_blocker_is_empty(bs_new));

    assert(bs_new->io_limits_enabled == false);

    assert(!throttle_have_timer(&bs_new->throttle_state));



    /* insert the nodes back into the graph node list if needed */

    if (bs_new->node_name[0] != '\0') {

        QTAILQ_INSERT_TAIL(&graph_bdrv_states, bs_new, node_list);

    }

    if (bs_old->node_name[0] != '\0') {

        QTAILQ_INSERT_TAIL(&graph_bdrv_states, bs_old, node_list);

    }



    bdrv_rebind(bs_new);

    bdrv_rebind(bs_old);

}
