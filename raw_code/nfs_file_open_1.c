static int nfs_file_open(BlockDriverState *bs, QDict *options, int flags,

                         Error **errp) {

    NFSClient *client = bs->opaque;

    int64_t ret;



    client->aio_context = bdrv_get_aio_context(bs);



    ret = nfs_client_open(client, options,

                          (flags & BDRV_O_RDWR) ? O_RDWR : O_RDONLY,

                          bs->open_flags, errp);

    if (ret < 0) {

        return ret;

    }

    qemu_mutex_init(&client->mutex);

    bs->total_sectors = ret;

    ret = 0;

    return ret;

}
