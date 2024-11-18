static void qobject_input_start_list(Visitor *v, const char *name,

                                     GenericList **list, size_t size,

                                     Error **errp)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qobject_input_get_object(qiv, name, true, errp);

    const QListEntry *entry;



    if (list) {

        *list = NULL;

    }

    if (!qobj) {

        return;

    }

    if (qobject_type(qobj) != QTYPE_QLIST) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "list");

        return;

    }



    entry = qobject_input_push(qiv, qobj, list);

    if (entry && list) {

        *list = g_malloc0(size);

    }

}
