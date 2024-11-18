const char *bdrv_get_format_name(BlockDriverState *bs)

{

    return bs->drv ? bs->drv->format_name : NULL;

}
