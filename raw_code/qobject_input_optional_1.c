static void qobject_input_optional(Visitor *v, const char *name, bool *present)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qobject_input_get_object(qiv, name, false, NULL);



    if (!qobj) {

        *present = false;

        return;

    }



    *present = true;

}
