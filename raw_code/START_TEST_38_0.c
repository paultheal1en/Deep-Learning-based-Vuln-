START_TEST(qstring_from_str_test)

{

    QString *qstring;

    const char *str = "QEMU";



    qstring = qstring_from_str(str);

    fail_unless(qstring != NULL);

    fail_unless(qstring->base.refcnt == 1);

    fail_unless(strcmp(str, qstring->string) == 0);

    fail_unless(qobject_type(QOBJECT(qstring)) == QTYPE_QSTRING);



    // destroy doesn't exit yet

    g_free(qstring->string);

    g_free(qstring);

}
