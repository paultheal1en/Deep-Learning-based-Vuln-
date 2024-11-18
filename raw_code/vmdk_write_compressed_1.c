static int vmdk_write_compressed(BlockDriverState *bs,

                                 int64_t sector_num,

                                 const uint8_t *buf,

                                 int nb_sectors)

{

    BDRVVmdkState *s = bs->opaque;



    if (s->num_extents == 1 && s->extents[0].compressed) {

        Coroutine *co;

        AioContext *aio_context = bdrv_get_aio_context(bs);

        VmdkWriteCompressedCo data = {

            .bs         = bs,

            .sector_num = sector_num,

            .buf        = buf,

            .nb_sectors = nb_sectors,

            .ret        = -EINPROGRESS,

        };

        co = qemu_coroutine_create(vmdk_co_write_compressed);

        qemu_coroutine_enter(co, &data);

        while (data.ret == -EINPROGRESS) {

            aio_poll(aio_context, true);

        }

        return data.ret;

    } else {

        return -ENOTSUP;

    }

}
