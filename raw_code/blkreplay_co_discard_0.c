static int coroutine_fn blkreplay_co_discard(BlockDriverState *bs,

    int64_t sector_num, int nb_sectors)

{

    uint64_t reqid = request_id++;

    int ret = bdrv_co_discard(bs->file->bs, sector_num, nb_sectors);

    block_request_create(reqid, bs, qemu_coroutine_self());

    qemu_coroutine_yield();



    return ret;

}
