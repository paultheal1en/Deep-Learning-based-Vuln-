NBDExport *nbd_export_new(BlockDriverState *bs, off_t dev_offset,

                          off_t size, uint32_t nbdflags,

                          void (*close)(NBDExport *))

{

    NBDExport *exp = g_malloc0(sizeof(NBDExport));

    exp->refcount = 1;

    QTAILQ_INIT(&exp->clients);

    exp->bs = bs;

    exp->dev_offset = dev_offset;

    exp->nbdflags = nbdflags;

    exp->size = size == -1 ? bdrv_getlength(bs) : size;

    exp->close = close;

    exp->ctx = bdrv_get_aio_context(bs);

    bdrv_ref(bs);

    bdrv_add_aio_context_notifier(bs, bs_aio_attached, bs_aio_detach, exp);







    return exp;

}