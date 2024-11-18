static void qobject_input_check_struct(Visitor *v, Error **errp)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    StackObject *tos = QSLIST_FIRST(&qiv->stack);



    assert(tos && !tos->entry);

    if (qiv->strict) {

        GHashTable *const top_ht = tos->h;

        if (top_ht) {

            GHashTableIter iter;

            const char *key;



            g_hash_table_iter_init(&iter, top_ht);

            if (g_hash_table_iter_next(&iter, (void **)&key, NULL)) {

                error_setg(errp, "Parameter '%s' is unexpected", key);

            }

        }

    }

}
