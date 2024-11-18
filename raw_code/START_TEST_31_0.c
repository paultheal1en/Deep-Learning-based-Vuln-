START_TEST(qint_destroy_test)

{

    QInt *qi = qint_from_int(0);

    QDECREF(qi);

}
