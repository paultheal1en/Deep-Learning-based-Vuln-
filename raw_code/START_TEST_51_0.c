START_TEST(unterminated_dict)

{

    QObject *obj = qobject_from_json("{'abc':32");

    fail_unless(obj == NULL);

}
