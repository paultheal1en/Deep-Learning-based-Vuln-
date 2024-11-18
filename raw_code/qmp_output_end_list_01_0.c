static void qmp_output_end_list(Visitor *v)

{

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_pop(qov);

}
