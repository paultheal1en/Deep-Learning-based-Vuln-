static void vararg_number(void)

{

    QObject *obj;

    QInt *qint;

    QFloat *qfloat;

    int value = 0x2342;

    int64_t value64 = 0x2342342343LL;

    double valuef = 2.323423423;



    obj = qobject_from_jsonf("%d", value);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QINT);



    qint = qobject_to_qint(obj);

    g_assert(qint_get_int(qint) == value);



    QDECREF(qint);



    obj = qobject_from_jsonf("%" PRId64, value64);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QINT);



    qint = qobject_to_qint(obj);

    g_assert(qint_get_int(qint) == value64);



    QDECREF(qint);



    obj = qobject_from_jsonf("%f", valuef);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QFLOAT);



    qfloat = qobject_to_qfloat(obj);

    g_assert(qfloat_get_double(qfloat) == valuef);



    QDECREF(qfloat);

}
