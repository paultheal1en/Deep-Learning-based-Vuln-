static void dump_json_image_info(ImageInfo *info)

{

    QString *str;

    QObject *obj;

    Visitor *v = qmp_output_visitor_new(&obj);



    visit_type_ImageInfo(v, NULL, &info, &error_abort);

    visit_complete(v, &obj);

    str = qobject_to_json_pretty(obj);

    assert(str != NULL);

    printf("%s\n", qstring_get_str(str));

    qobject_decref(obj);

    visit_free(v);

    QDECREF(str);

}
