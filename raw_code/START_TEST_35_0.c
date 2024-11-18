START_TEST(qfloat_from_double_test)

{

    QFloat *qf;

    const double value = -42.23423;



    qf = qfloat_from_double(value);

    fail_unless(qf != NULL);

    fail_unless(qf->value == value);

    fail_unless(qf->base.refcnt == 1);

    fail_unless(qobject_type(QOBJECT(qf)) == QTYPE_QFLOAT);



    // destroy doesn't exit yet

    g_free(qf);

}
