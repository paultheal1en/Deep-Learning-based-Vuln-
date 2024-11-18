void blk_remove_bs(BlockBackend *blk)

{

    assert(blk->root->bs->blk == blk);



    notifier_list_notify(&blk->remove_bs_notifiers, blk);

    if (blk->public.throttle_state) {

        throttle_timers_detach_aio_context(&blk->public.throttle_timers);

    }



    blk_update_root_state(blk);



    blk->root->bs->blk = NULL;

    bdrv_root_unref_child(blk->root);

    blk->root = NULL;

}
