static void qfloat_destroy_obj(QObject *obj)

{

    assert(obj != NULL);

    g_free(qobject_to_qfloat(obj));

}
