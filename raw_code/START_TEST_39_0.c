START_TEST(qdict_get_test)

{

    QInt *qi;

    QObject *obj;

    const int value = -42;

    const char *key = "test";



    qdict_put(tests_dict, key, qint_from_int(value));



    obj = qdict_get(tests_dict, key);

    fail_unless(obj != NULL);



    qi = qobject_to_qint(obj);

    fail_unless(qint_get_int(qi) == value);

}
