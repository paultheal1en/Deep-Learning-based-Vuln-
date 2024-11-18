const char *bdrv_get_device_or_node_name(const BlockDriverState *bs)

{

    return bs->blk ? blk_name(bs->blk) : bs->node_name;

}
