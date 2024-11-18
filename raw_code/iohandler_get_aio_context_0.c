AioContext *iohandler_get_aio_context(void)

{

    iohandler_init();

    return iohandler_ctx;

}
