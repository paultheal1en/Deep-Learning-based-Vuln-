static void qmp_input_push(QmpInputVisitor *qiv, QObject *obj, Error **errp)

{

    GHashTable *h;

    StackObject *tos = &qiv->stack[qiv->nb_stack];



    assert(obj);

    if (qiv->nb_stack >= QIV_STACK_SIZE) {

        error_setg(errp, "An internal buffer overran");

        return;

    }



    tos->obj = obj;

    assert(!tos->h);

    assert(!tos->entry);



    if (qiv->strict && qobject_type(obj) == QTYPE_QDICT) {

        h = g_hash_table_new(g_str_hash, g_str_equal);

        qdict_iter(qobject_to_qdict(obj), qdict_add_key, h);

        tos->h = h;

    } else if (qobject_type(obj) == QTYPE_QLIST) {

        tos->entry = qlist_first(qobject_to_qlist(obj));

        tos->first = true;

    }



    qiv->nb_stack++;

}
