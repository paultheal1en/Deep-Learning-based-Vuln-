static void iohandler_init(void)

{

    if (!iohandler_ctx) {

        iohandler_ctx = aio_context_new(&error_abort);

    }

}
