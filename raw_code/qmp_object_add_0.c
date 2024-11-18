void qmp_object_add(const char *type, const char *id,

                    bool has_props, QObject *props, Error **errp)

{

    const QDict *pdict = NULL;

    QmpInputVisitor *qiv;

    Object *obj;



    if (props) {

        pdict = qobject_to_qdict(props);

        if (!pdict) {

            error_setg(errp, QERR_INVALID_PARAMETER_TYPE, "props", "dict");

            return;

        }

    }



    qiv = qmp_input_visitor_new(props, false);

    obj = user_creatable_add_type(type, id, pdict,

                                  qmp_input_get_visitor(qiv), errp);

    qmp_input_visitor_cleanup(qiv);

    if (obj) {

        object_unref(obj);

    }

}
