void aio_context_set_poll_params(AioContext *ctx, int64_t max_ns, Error **errp)

{

    error_setg(errp, "AioContext polling is not implemented on Windows");

}
