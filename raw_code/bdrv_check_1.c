int bdrv_check(BlockDriverState *bs, BdrvCheckResult *res)

{

    if (bs->drv->bdrv_check == NULL) {

        return -ENOTSUP;

    }



    memset(res, 0, sizeof(*res));

    res->corruptions = bs->drv->bdrv_check(bs);

    return res->corruptions < 0 ? res->corruptions : 0;

}
