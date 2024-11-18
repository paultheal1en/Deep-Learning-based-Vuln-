static void qmp_input_start_struct(Visitor *v, const char *name, void **obj,

                                   size_t size, Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true, errp);

    Error *err = NULL;



    if (obj) {

        *obj = NULL;

    }

    if (!qobj) {

        return;

    }

    if (qobject_type(qobj) != QTYPE_QDICT) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "QDict");

        return;

    }



    qmp_input_push(qiv, qobj, obj, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    if (obj) {

        *obj = g_malloc0(size);

    }

}
