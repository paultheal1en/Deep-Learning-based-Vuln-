static InputEvent *qapi_clone_InputEvent(InputEvent *src)

{

    QmpOutputVisitor *qov;

    QmpInputVisitor *qiv;

    Visitor *ov, *iv;

    QObject *obj;

    InputEvent *dst = NULL;



    qov = qmp_output_visitor_new();

    ov = qmp_output_get_visitor(qov);

    visit_type_InputEvent(ov, NULL, &src, &error_abort);

    obj = qmp_output_get_qobject(qov);

    qmp_output_visitor_cleanup(qov);

    if (!obj) {

        return NULL;

    }



    qiv = qmp_input_visitor_new(obj, false);

    iv = qmp_input_get_visitor(qiv);

    visit_type_InputEvent(iv, NULL, &dst, &error_abort);

    qmp_input_visitor_cleanup(qiv);

    qobject_decref(obj);



    return dst;

}
