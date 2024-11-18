static void qmp_input_type_any(Visitor *v, const char *name, QObject **obj,
                               Error **errp)
{
    QmpInputVisitor *qiv = to_qiv(v);
    QObject *qobj = qmp_input_get_object(qiv, name, true);
    qobject_incref(qobj);
    *obj = qobj;