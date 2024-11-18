static void qmp_input_start_list(Visitor *v, const char *name,

                                 GenericList **list, size_t size, Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true, errp);

    const QListEntry *entry;



    if (!qobj) {

        return;

    }

    if (qobject_type(qobj) != QTYPE_QLIST) {

        if (list) {

            *list = NULL;

        }

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "list");

        return;

    }



    entry = qmp_input_push(qiv, qobj, list, errp);

    if (list) {

        if (entry) {

            *list = g_malloc0(size);

        } else {

            *list = NULL;

        }

    }

}
