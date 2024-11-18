static int iscsi_create(const char *filename, QEMUOptionParameter *options)

{

    int ret = 0;

    int64_t total_size = 0;

    BlockDriverState bs;

    IscsiLun *iscsilun = NULL;

    QDict *bs_options;



    memset(&bs, 0, sizeof(BlockDriverState));



    /* Read out options */

    while (options && options->name) {

        if (!strcmp(options->name, "size")) {

            total_size = options->value.n / BDRV_SECTOR_SIZE;

        }

        options++;

    }



    bs.opaque = g_malloc0(sizeof(struct IscsiLun));

    iscsilun = bs.opaque;



    bs_options = qdict_new();

    qdict_put(bs_options, "filename", qstring_from_str(filename));

    ret = iscsi_open(&bs, bs_options, 0);

    QDECREF(bs_options);



    if (ret != 0) {

        goto out;

    }

    if (iscsilun->nop_timer) {

        timer_del(iscsilun->nop_timer);

        timer_free(iscsilun->nop_timer);

    }

    if (iscsilun->type != TYPE_DISK) {

        ret = -ENODEV;

        goto out;

    }

    if (bs.total_sectors < total_size) {

        ret = -ENOSPC;

        goto out;

    }



    ret = 0;

out:

    if (iscsilun->iscsi != NULL) {

        iscsi_destroy_context(iscsilun->iscsi);

    }

    g_free(bs.opaque);

    return ret;

}
