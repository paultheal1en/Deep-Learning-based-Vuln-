static void bdrv_drain_recurse(BlockDriverState *bs)

{

    BdrvChild *child;



    if (bs->drv && bs->drv->bdrv_drain) {

        bs->drv->bdrv_drain(bs);

    }

    QLIST_FOREACH(child, &bs->children, next) {

        bdrv_drain_recurse(child->bs);

    }

}
