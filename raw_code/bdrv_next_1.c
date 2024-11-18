BdrvNextIterator *bdrv_next(BdrvNextIterator *it, BlockDriverState **bs)

{

    if (!it) {

        it = g_new(BdrvNextIterator, 1);

        *it = (BdrvNextIterator) {

            .phase = BDRV_NEXT_BACKEND_ROOTS,

        };

    }



    /* First, return all root nodes of BlockBackends. In order to avoid

     * returning a BDS twice when multiple BBs refer to it, we only return it

     * if the BB is the first one in the parent list of the BDS. */

    if (it->phase == BDRV_NEXT_BACKEND_ROOTS) {

        do {

            it->blk = blk_all_next(it->blk);

            *bs = it->blk ? blk_bs(it->blk) : NULL;

        } while (it->blk && (*bs == NULL || bdrv_first_blk(*bs) != it->blk));



        if (*bs) {

            return it;

        }

        it->phase = BDRV_NEXT_MONITOR_OWNED;

    }



    /* Then return the monitor-owned BDSes without a BB attached. Ignore all

     * BDSes that are attached to a BlockBackend here; they have been handled

     * by the above block already */

    do {

        it->bs = bdrv_next_monitor_owned(it->bs);

        *bs = it->bs;

    } while (*bs && bdrv_has_blk(*bs));



    return *bs ? it : NULL;

}
