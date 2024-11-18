static void qmp_output_end_struct(Visitor *v, void **obj)

{

    QmpOutputVisitor *qov = to_qov(v);

    QObject *value = qmp_output_pop(qov, obj);

    assert(qobject_type(value) == QTYPE_QDICT);

}
