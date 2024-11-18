static void qobject_input_type_number(Visitor *v, const char *name, double *obj,

                                      Error **errp)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qobject_input_get_object(qiv, name, true, errp);

    QInt *qint;

    QFloat *qfloat;



    if (!qobj) {

        return;

    }

    qint = qobject_to_qint(qobj);

    if (qint) {

        *obj = qint_get_int(qobject_to_qint(qobj));

        return;

    }



    qfloat = qobject_to_qfloat(qobj);

    if (qfloat) {

        *obj = qfloat_get_double(qobject_to_qfloat(qobj));

        return;

    }



    error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

               "number");

}
