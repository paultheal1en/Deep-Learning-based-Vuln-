BlockDriverState *bdrv_new(const char *device_name)

{

    BlockDriverState *bs;



    bs = g_malloc0(sizeof(BlockDriverState));

    QLIST_INIT(&bs->dirty_bitmaps);

    pstrcpy(bs->device_name, sizeof(bs->device_name), device_name);

    if (device_name[0] != '\0') {

        QTAILQ_INSERT_TAIL(&bdrv_states, bs, device_list);

    }

    bdrv_iostatus_disable(bs);

    notifier_list_init(&bs->close_notifiers);

    notifier_with_return_list_init(&bs->before_write_notifiers);

    qemu_co_queue_init(&bs->throttled_reqs[0]);

    qemu_co_queue_init(&bs->throttled_reqs[1]);

    bs->refcnt = 1;



    return bs;

}
