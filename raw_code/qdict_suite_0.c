static Suite *qdict_suite(void)

{

    Suite *s;

    TCase *qdict_public_tcase;

    TCase *qdict_public2_tcase;

    TCase *qdict_stress_tcase;

    TCase *qdict_errors_tcase;



    s = suite_create("QDict test-suite");



    qdict_public_tcase = tcase_create("Public Interface");

    suite_add_tcase(s, qdict_public_tcase);

    tcase_add_test(qdict_public_tcase, qdict_new_test);

    tcase_add_test(qdict_public_tcase, qdict_put_obj_test);

    tcase_add_test(qdict_public_tcase, qdict_destroy_simple_test);



    /* Continue, but now with fixtures */

    qdict_public2_tcase = tcase_create("Public Interface (2)");

    suite_add_tcase(s, qdict_public2_tcase);

    tcase_add_checked_fixture(qdict_public2_tcase, qdict_setup, qdict_teardown);

    tcase_add_test(qdict_public2_tcase, qdict_get_test);

    tcase_add_test(qdict_public2_tcase, qdict_get_int_test);

    tcase_add_test(qdict_public2_tcase, qdict_get_try_int_test);

    tcase_add_test(qdict_public2_tcase, qdict_get_str_test);

    tcase_add_test(qdict_public2_tcase, qdict_get_try_str_test);

    tcase_add_test(qdict_public2_tcase, qdict_haskey_not_test);

    tcase_add_test(qdict_public2_tcase, qdict_haskey_test);

    tcase_add_test(qdict_public2_tcase, qdict_del_test);

    tcase_add_test(qdict_public2_tcase, qobject_to_qdict_test);

    tcase_add_test(qdict_public2_tcase, qdict_iterapi_test);



    qdict_errors_tcase = tcase_create("Errors");

    suite_add_tcase(s, qdict_errors_tcase);

    tcase_add_checked_fixture(qdict_errors_tcase, qdict_setup, qdict_teardown);

    tcase_add_test(qdict_errors_tcase, qdict_put_exists_test);

    tcase_add_test(qdict_errors_tcase, qdict_get_not_exists_test);



    /* The Big one */

    qdict_stress_tcase = tcase_create("Stress Test");

    suite_add_tcase(s, qdict_stress_tcase);

    tcase_add_test(qdict_stress_tcase, qdict_stress_test);



    return s;

}
