START_TEST(unterminated_sq_string)

{

    QObject *obj = qobject_from_json("'abc");

    fail_unless(obj == NULL);

}
