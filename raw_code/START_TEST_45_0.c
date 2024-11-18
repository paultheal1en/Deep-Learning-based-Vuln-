START_TEST(vararg_number)

{

    QObject *obj;

    QInt *qint;

    QFloat *qfloat;

    int value = 0x2342;

    int64_t value64 = 0x2342342343LL;

    double valuef = 2.323423423;



    obj = qobject_from_jsonf("%d", value);

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QINT);



    qint = qobject_to_qint(obj);

    fail_unless(qint_get_int(qint) == value);



    QDECREF(qint);



    obj = qobject_from_jsonf("%" PRId64, value64);

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QINT);



    qint = qobject_to_qint(obj);

    fail_unless(qint_get_int(qint) == value64);



    QDECREF(qint);



    obj = qobject_from_jsonf("%f", valuef);

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QFLOAT);



    qfloat = qobject_to_qfloat(obj);

    fail_unless(qfloat_get_double(qfloat) == valuef);



    QDECREF(qfloat);

}
