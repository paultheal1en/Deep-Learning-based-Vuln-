static Suite *qstring_suite(void)

{

    Suite *s;

    TCase *qstring_public_tcase;



    s = suite_create("QString test-suite");



    qstring_public_tcase = tcase_create("Public Interface");

    suite_add_tcase(s, qstring_public_tcase);

    tcase_add_test(qstring_public_tcase, qstring_from_str_test);

    tcase_add_test(qstring_public_tcase, qstring_destroy_test);

    tcase_add_test(qstring_public_tcase, qstring_get_str_test);

    tcase_add_test(qstring_public_tcase, qstring_append_chr_test);

    tcase_add_test(qstring_public_tcase, qstring_from_substr_test);

    tcase_add_test(qstring_public_tcase, qobject_to_qstring_test);



    return s;

}
