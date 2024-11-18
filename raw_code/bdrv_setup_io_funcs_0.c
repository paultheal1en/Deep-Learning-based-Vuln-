void bdrv_setup_io_funcs(BlockDriver *bdrv)

{

    /* Block drivers without coroutine functions need emulation */

    if (!bdrv->bdrv_co_readv) {

        bdrv->bdrv_co_readv = bdrv_co_readv_em;

        bdrv->bdrv_co_writev = bdrv_co_writev_em;



        /* bdrv_co_readv_em()/brdv_co_writev_em() work in terms of aio, so if

         * the block driver lacks aio we need to emulate that too.

         */

        if (!bdrv->bdrv_aio_readv) {

            /* add AIO emulation layer */

            bdrv->bdrv_aio_readv = bdrv_aio_readv_em;

            bdrv->bdrv_aio_writev = bdrv_aio_writev_em;

        }

    }

}
