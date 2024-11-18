START_TEST(qobject_to_qstring_test)

{

    QString *qstring;



    qstring = qstring_from_str("foo");

    fail_unless(qobject_to_qstring(QOBJECT(qstring)) == qstring);



    QDECREF(qstring);

}
