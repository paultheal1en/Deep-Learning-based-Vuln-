static void dump_json_image_info(ImageInfo *info)

{

    Error *local_err = NULL;

    QString *str;

    QmpOutputVisitor *ov = qmp_output_visitor_new();

    QObject *obj;

    visit_type_ImageInfo(qmp_output_get_visitor(ov), NULL, &info, &local_err);

    obj = qmp_output_get_qobject(ov);

    str = qobject_to_json_pretty(obj);

    assert(str != NULL);

    printf("%s\n", qstring_get_str(str));

    qobject_decref(obj);

    qmp_output_visitor_cleanup(ov);

    QDECREF(str);

}
