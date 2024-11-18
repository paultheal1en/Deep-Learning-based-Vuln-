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

        co = qemu_coroutine_create(bdrv_pdiscard_co_entry, &rwco);

        qemu_coroutine_enter(co);

        BDRV_POLL_WHILE(bs, rwco.ret == NOT_DONE);

    }



    return rwco.ret;

}
