int bdrv_flush(BlockDriverState *bs)

{

    Coroutine *co;

    RwCo rwco = {

        .bs = bs,

        .ret = NOT_DONE,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_flush_co_entry(&rwco);

    } else {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        co = qemu_coroutine_create(bdrv_flush_co_entry);

        qemu_coroutine_enter(co, &rwco);

        while (rwco.ret == NOT_DONE) {

            aio_poll(aio_context, true);

        }

    }



    return rwco.ret;

}
