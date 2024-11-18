START_TEST(empty_input)

{

    const char *empty = "";



    QObject *obj = qobject_from_json(empty);

    fail_unless(obj == NULL);

}
