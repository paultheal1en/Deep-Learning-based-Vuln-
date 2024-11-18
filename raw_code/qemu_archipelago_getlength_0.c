static int64_t qemu_archipelago_getlength(BlockDriverState *bs)

{

    int64_t ret;

    BDRVArchipelagoState *s = bs->opaque;



    ret = archipelago_volume_info(s);

    return ret;

}
