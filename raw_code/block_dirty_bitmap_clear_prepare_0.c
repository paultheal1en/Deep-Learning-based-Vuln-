static void block_dirty_bitmap_clear_prepare(BlkActionState *common,

                                             Error **errp)

{

    BlockDirtyBitmapState *state = DO_UPCAST(BlockDirtyBitmapState,

                                             common, common);

    BlockDirtyBitmap *action;



    if (action_check_completion_mode(common, errp) < 0) {

        return;

    }



    action = common->action->u.block_dirty_bitmap_clear.data;

    state->bitmap = block_dirty_bitmap_lookup(action->node,

                                              action->name,

                                              &state->bs,

                                              &state->aio_context,

                                              errp);

    if (!state->bitmap) {

        return;

    }



    if (bdrv_dirty_bitmap_frozen(state->bitmap)) {

        error_setg(errp, "Cannot modify a frozen bitmap");

        return;

    } else if (!bdrv_dirty_bitmap_enabled(state->bitmap)) {

        error_setg(errp, "Cannot clear a disabled bitmap");

        return;

    }



    bdrv_clear_dirty_bitmap(state->bitmap, &state->backup);

    /* AioContext is released in .clean() */

}
