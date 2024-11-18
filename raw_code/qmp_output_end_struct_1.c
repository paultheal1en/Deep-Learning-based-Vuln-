static void qmp_output_end_struct(Visitor *v, Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    QObject *value = qmp_output_pop(qov);

    assert(qobject_type(value) == QTYPE_QDICT);

}
