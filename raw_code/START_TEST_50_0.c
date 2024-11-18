START_TEST(qint_from_int64_test)

{

    QInt *qi;

    const int64_t value = 0x1234567890abcdefLL;



    qi = qint_from_int(value);

    fail_unless((int64_t) qi->value == value);



    QDECREF(qi);

}
