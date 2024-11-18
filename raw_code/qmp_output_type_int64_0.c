static void qmp_output_type_int64(Visitor *v, const char *name, int64_t *obj,

                                  Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_add(qov, name, qint_from_int(*obj));

}
