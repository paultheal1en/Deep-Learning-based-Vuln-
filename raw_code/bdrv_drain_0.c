void bdrv_drain(BlockDriverState *bs)

{

    while (bdrv_drain_one(bs)) {

        /* Keep iterating */

    }

}
