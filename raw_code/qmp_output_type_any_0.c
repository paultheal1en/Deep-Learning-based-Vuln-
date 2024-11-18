static void qmp_output_type_any(Visitor *v, const char *name, QObject **obj,

                                Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    qobject_incref(*obj);

    qmp_output_add_obj(qov, name, *obj);

}
