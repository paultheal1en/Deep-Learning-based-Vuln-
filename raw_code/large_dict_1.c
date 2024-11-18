static void large_dict(void)

{

    GString *gstr = g_string_new("");

    QObject *obj;



    gen_test_json(gstr, 10, 100);

    obj = qobject_from_json(gstr->str, NULL);

    g_assert(obj != NULL);



    qobject_decref(obj);

    g_string_free(gstr, true);

}
