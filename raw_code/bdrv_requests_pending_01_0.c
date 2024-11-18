bool bdrv_requests_pending(BlockDriverState *bs)

{

    BdrvChild *child;

    BlockBackendPublic *blkp = bs->blk ? blk_get_public(bs->blk) : NULL;



    if (!QLIST_EMPTY(&bs->tracked_requests)) {

        return true;

    }

    if (blkp && !qemu_co_queue_empty(&blkp->throttled_reqs[0])) {

        return true;

    }

    if (blkp && !qemu_co_queue_empty(&blkp->throttled_reqs[1])) {

        return true;

    }



    QLIST_FOREACH(child, &bs->children, next) {

        if (bdrv_requests_pending(child->bs)) {

            return true;

        }

    }



    return false;

}
