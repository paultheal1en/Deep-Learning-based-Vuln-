static int coroutine_fn blkreplay_co_pdiscard(BlockDriverState *bs,

                                              int64_t offset, int count)

{

    uint64_t reqid = request_id++;

    int ret = bdrv_co_pdiscard(bs->file->bs, offset, count);

    block_request_create(reqid, bs, qemu_coroutine_self());

    qemu_coroutine_yield();



    return ret;

}
