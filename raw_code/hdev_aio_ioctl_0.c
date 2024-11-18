static BlockDriverAIOCB *hdev_aio_ioctl(BlockDriverState *bs,

        unsigned long int req, void *buf,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    BDRVRawState *s = bs->opaque;



    if (fd_open(bs) < 0)

        return NULL;

    return paio_ioctl(bs, s->fd, req, buf, cb, opaque);

}
