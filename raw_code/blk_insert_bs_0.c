void blk_insert_bs(BlockBackend *blk, BlockDriverState *bs)

{

    bdrv_ref(bs);

    blk->root = bdrv_root_attach_child(bs, "root", &child_root, blk);



    notifier_list_notify(&blk->insert_bs_notifiers, blk);

    if (blk->public.throttle_state) {

        throttle_timers_attach_aio_context(

            &blk->public.throttle_timers, bdrv_get_aio_context(bs));

    }

}
