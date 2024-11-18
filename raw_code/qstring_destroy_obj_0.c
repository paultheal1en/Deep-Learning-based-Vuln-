static void qstring_destroy_obj(QObject *obj)

{

    QString *qs;



    assert(obj != NULL);

    qs = qobject_to_qstring(obj);

    g_free(qs->string);

    g_free(qs);

}
