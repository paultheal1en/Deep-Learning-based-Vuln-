int bdrv_flush_all(void)

{

    BlockDriverState *bs = NULL;

    int result = 0;



    while ((bs = bdrv_next(bs))) {

        AioContext *aio_context = bdrv_get_aio_context(bs);

        int ret;



        aio_context_acquire(aio_context);

        ret = bdrv_flush(bs);

        if (ret < 0 && !result) {

            result = ret;

        }

        aio_context_release(aio_context);

    }



    return result;

}
