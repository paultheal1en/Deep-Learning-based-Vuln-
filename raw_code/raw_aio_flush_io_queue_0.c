static void raw_aio_flush_io_queue(BlockDriverState *bs)

{

#ifdef CONFIG_LINUX_AIO

    BDRVRawState *s = bs->opaque;

    if (s->use_aio) {

        laio_io_unplug(bs, s->aio_ctx, false);

    }

#endif

}
