static void iscsi_close(BlockDriverState *bs)

{

    IscsiLun *iscsilun = bs->opaque;

    struct iscsi_context *iscsi = iscsilun->iscsi;



    iscsi_detach_aio_context(bs);

    if (iscsi_is_logged_in(iscsi)) {

        iscsi_logout_sync(iscsi);

    }

    iscsi_destroy_context(iscsi);

    g_free(iscsilun->zeroblock);

    g_free(iscsilun->allocationmap);

    memset(iscsilun, 0, sizeof(IscsiLun));

}
