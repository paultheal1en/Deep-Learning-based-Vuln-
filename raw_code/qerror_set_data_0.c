static void qerror_set_data(QError *qerr, const char *fmt, va_list *va)

{

    QObject *obj;



    obj = qobject_from_jsonv(fmt, va);

    if (!obj) {

        qerror_abort(qerr, "invalid format '%s'", fmt);

    }

    if (qobject_type(obj) != QTYPE_QDICT) {

        qerror_abort(qerr, "error format is not a QDict '%s'", fmt);

    }



    qerr->error = qobject_to_qdict(obj);



    obj = qdict_get(qerr->error, "class");

    if (!obj) {

        qerror_abort(qerr, "missing 'class' key in '%s'", fmt);

    }

    if (qobject_type(obj) != QTYPE_QSTRING) {

        qerror_abort(qerr, "'class' key value should be a QString");

    }

    

    obj = qdict_get(qerr->error, "data");

    if (!obj) {

        qerror_abort(qerr, "missing 'data' key in '%s'", fmt);

    }

    if (qobject_type(obj) != QTYPE_QDICT) {

        qerror_abort(qerr, "'data' key value should be a QDICT");

    }

}
