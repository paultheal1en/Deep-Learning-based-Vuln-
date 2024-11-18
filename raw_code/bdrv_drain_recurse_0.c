static bool bdrv_drain_recurse(BlockDriverState *bs)

{

    BdrvChild *child;

    bool waited;



    waited = bdrv_drain_poll(bs);



    if (bs->drv && bs->drv->bdrv_drain) {

        bs->drv->bdrv_drain(bs);

    }



    QLIST_FOREACH(child, &bs->children, next) {

        waited |= bdrv_drain_recurse(child->bs);

    }



    return waited;

}
