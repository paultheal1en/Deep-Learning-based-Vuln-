static void qmp_input_pop(QmpInputVisitor *qiv, Error **errp)

{

    StackObject *tos = &qiv->stack[qiv->nb_stack - 1];

    assert(qiv->nb_stack > 0);



    if (qiv->strict) {

        GHashTable *const top_ht = tos->h;

        if (top_ht) {

            GHashTableIter iter;

            const char *key;



            g_hash_table_iter_init(&iter, top_ht);

            if (g_hash_table_iter_next(&iter, (void **)&key, NULL)) {

                error_setg(errp, QERR_QMP_EXTRA_MEMBER, key);

            }

            g_hash_table_unref(top_ht);

        }

        tos->h = NULL;

    }



    qiv->nb_stack--;

}
