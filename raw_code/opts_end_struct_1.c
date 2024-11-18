opts_end_struct(Visitor *v, Error **errp)

{

    OptsVisitor *ov = to_ov(v);

    GHashTableIter iter;

    GQueue *any;



    if (--ov->depth > 0) {

        return;

    }



    /* we should have processed all (distinct) QemuOpt instances */

    g_hash_table_iter_init(&iter, ov->unprocessed_opts);

    if (g_hash_table_iter_next(&iter, NULL, (void **)&any)) {

        const QemuOpt *first;



        first = g_queue_peek_head(any);

        error_setg(errp, QERR_INVALID_PARAMETER, first->name);

    }

    g_hash_table_destroy(ov->unprocessed_opts);

    ov->unprocessed_opts = NULL;

    if (ov->fake_id_opt) {

        g_free(ov->fake_id_opt->name);

        g_free(ov->fake_id_opt->str);

        g_free(ov->fake_id_opt);

    }

    ov->fake_id_opt = NULL;

}
