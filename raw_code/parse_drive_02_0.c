static int parse_drive(DeviceState *dev, const char *str, void **ptr)

{

    BlockDriverState *bs;



    bs = bdrv_find(str);

    if (bs == NULL) {

        return -ENOENT;

    }

    if (bdrv_attach_dev(bs, dev) < 0) {

        return -EEXIST;

    }

    *ptr = bs;

    return 0;

}
