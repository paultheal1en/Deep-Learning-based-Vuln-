START_TEST(qint_from_int_test)

{

    QInt *qi;

    const int value = -42;



    qi = qint_from_int(value);

    fail_unless(qi != NULL);

    fail_unless(qi->value == value);

    fail_unless(qi->base.refcnt == 1);

    fail_unless(qobject_type(QOBJECT(qi)) == QTYPE_QINT);



    // destroy doesn't exit yet

    g_free(qi);

}
