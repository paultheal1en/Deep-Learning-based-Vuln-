BlockDriverState *bdrv_next(BlockDriverState *bs)

{

    if (!bs) {

        return QTAILQ_FIRST(&bdrv_states);

    }

    return QTAILQ_NEXT(bs, device_list);

}
