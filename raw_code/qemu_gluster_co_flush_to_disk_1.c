static coroutine_fn int qemu_gluster_co_flush_to_disk(BlockDriverState *bs)

{

    int ret;

    GlusterAIOCB acb;

    BDRVGlusterState *s = bs->opaque;



    acb.size = 0;

    acb.ret = 0;

    acb.coroutine = qemu_coroutine_self();

    acb.aio_context = bdrv_get_aio_context(bs);



    ret = glfs_fsync_async(s->fd, gluster_finish_aiocb, &acb);

    if (ret < 0) {

        return -errno;

    }



    qemu_coroutine_yield();

    return acb.ret;

}
