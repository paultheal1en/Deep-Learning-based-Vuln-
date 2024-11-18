void nbd_export_close(NBDExport *exp)

{

    NBDClient *client, *next;



    nbd_export_get(exp);

    QTAILQ_FOREACH_SAFE(client, &exp->clients, next, next) {

        client_close(client);

    }

    nbd_export_set_name(exp, NULL);

    nbd_export_put(exp);

    if (exp->blk) {

        blk_remove_aio_context_notifier(exp->blk, blk_aio_attached,

                                        blk_aio_detach, exp);

        blk_unref(exp->blk);

        exp->blk = NULL;

    }

}
