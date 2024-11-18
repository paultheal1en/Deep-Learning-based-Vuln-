static void dump_json_image_info_list(ImageInfoList *list)

{

    QString *str;

    QmpOutputVisitor *ov = qmp_output_visitor_new();

    QObject *obj;

    visit_type_ImageInfoList(qmp_output_get_visitor(ov), NULL, &list,

                             &error_abort);

    obj = qmp_output_get_qobject(ov);

    str = qobject_to_json_pretty(obj);

    assert(str != NULL);

    printf("%s\n", qstring_get_str(str));

    qobject_decref(obj);

    qmp_output_visitor_cleanup(ov);

    QDECREF(str);

}
