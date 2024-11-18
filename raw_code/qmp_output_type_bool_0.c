static void qmp_output_type_bool(Visitor *v, const char *name, bool *obj,

                                 Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_add(qov, name, qbool_from_bool(*obj));

}
