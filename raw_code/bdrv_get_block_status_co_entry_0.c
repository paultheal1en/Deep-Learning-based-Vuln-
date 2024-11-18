static void coroutine_fn bdrv_get_block_status_co_entry(void *opaque)

{

    BdrvCoGetBlockStatusData *data = opaque;

    BlockDriverState *bs = data->bs;



    data->ret = bdrv_co_get_block_status(bs, data->sector_num, data->nb_sectors,

                                         data->pnum);

    data->done = true;

}
