void iothread_stop_all(void)

{

    Object *container = object_get_objects_root();

    BlockDriverState *bs;

    BdrvNextIterator it;



    for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

        AioContext *ctx = bdrv_get_aio_context(bs);

        if (ctx == qemu_get_aio_context()) {

            continue;

        }

        aio_context_acquire(ctx);

        bdrv_set_aio_context(bs, qemu_get_aio_context());

        aio_context_release(ctx);

    }



    object_child_foreach(container, iothread_stop, NULL);

}
