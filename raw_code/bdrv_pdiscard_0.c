int bdrv_pdiscard(BlockDriverState *bs, int64_t offset, int count)

{

    Coroutine *co;

    DiscardCo rwco = {

        .bs = bs,

        .offset = offset,

        .count = count,

        .ret = NOT_DONE,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_pdiscard_co_entry(&rwco);

    } else {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        co = qemu_coroutine_create(bdrv_pdiscard_co_entry, &rwco);

        qemu_coroutine_enter(co);

        while (rwco.ret == NOT_DONE) {

            aio_poll(aio_context, true);

        }

    }



    return rwco.ret;

}
