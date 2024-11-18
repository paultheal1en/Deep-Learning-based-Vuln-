int main(int argc, char *argv[])

{

    g_test_init(&argc, &argv, NULL);



    qtest_add_func("qmp/protocol", test_qmp_protocol);



    return g_test_run();

}
