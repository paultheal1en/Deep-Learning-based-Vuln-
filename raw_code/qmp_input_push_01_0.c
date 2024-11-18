static const QListEntry *qmp_input_push(QmpInputVisitor *qiv, QObject *obj,

                                        void *qapi, Error **errp)

{

    GHashTable *h;

    StackObject *tos = g_new0(StackObject, 1);



    assert(obj);

    tos->obj = obj;

    tos->qapi = qapi;



    if (qiv->strict && qobject_type(obj) == QTYPE_QDICT) {

        h = g_hash_table_new(g_str_hash, g_str_equal);

        qdict_iter(qobject_to_qdict(obj), qdict_add_key, h);

        tos->h = h;

    } else if (qobject_type(obj) == QTYPE_QLIST) {

        tos->entry = qlist_first(qobject_to_qlist(obj));

    }



    QSLIST_INSERT_HEAD(&qiv->stack, tos, node);

    return tos->entry;

}
