static int raw_co_ioctl(BlockDriverState *bs, unsigned long int req, void *buf)

{

    BDRVRawState *s = bs->opaque;

    if (s->offset || s->has_size) {

        return -ENOTSUP;

    }

    return bdrv_co_ioctl(bs->file->bs, req, buf);

}
