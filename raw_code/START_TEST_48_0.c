START_TEST(qstring_append_chr_test)

{

    int i;

    QString *qstring;

    const char *str = "qstring append char unit-test";



    qstring = qstring_new();



    for (i = 0; str[i]; i++)

        qstring_append_chr(qstring, str[i]);



    fail_unless(strcmp(str, qstring_get_str(qstring)) == 0);

    QDECREF(qstring);

}
