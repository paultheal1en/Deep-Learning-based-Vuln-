int main(int argc, char *argv[])

{

    const char *sparc_machines[] = { "SPARCbook", "Voyager", "SS-20", NULL };

    const char *sparc64_machines[] = { "sun4u", "sun4v", NULL };

    const char *mac_machines[] = { "mac99", "g3beige", NULL };

    const char *arch = qtest_get_arch();



    g_test_init(&argc, &argv, NULL);



    if (!strcmp(arch, "ppc") || !strcmp(arch, "ppc64")) {

        add_tests(mac_machines);

    } else if (!strcmp(arch, "sparc")) {

        add_tests(sparc_machines);

    } else if (!strcmp(arch, "sparc64")) {

        add_tests(sparc64_machines);

    } else {

        g_assert_not_reached();

    }



    return g_test_run();

}
