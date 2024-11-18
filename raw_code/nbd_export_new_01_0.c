NBDExport *nbd_export_new(BlockDriverState *bs, off_t dev_offset,

                          off_t size, uint32_t nbdflags)

{

    NBDExport *exp = g_malloc0(sizeof(NBDExport));

    QSIMPLEQ_INIT(&exp->requests);

    exp->refcount = 1;

    QTAILQ_INIT(&exp->clients);

    exp->bs = bs;

    exp->dev_offset = dev_offset;

    exp->nbdflags = nbdflags;

    exp->size = size == -1 ? bdrv_getlength(bs) : size;

    return exp;

}
