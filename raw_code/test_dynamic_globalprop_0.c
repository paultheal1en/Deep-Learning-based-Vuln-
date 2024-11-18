static void test_dynamic_globalprop(void)

{

    g_test_trap_subprocess("/qdev/properties/dynamic/global/subprocess", 0, 0);

    g_test_trap_assert_passed();

    g_test_trap_assert_stderr_unmatched("*prop1*");

    g_test_trap_assert_stderr_unmatched("*prop2*");

    g_test_trap_assert_stderr("*Warning: \"-global dynamic-prop-type-bad.prop3=103\" not used\n*");

    g_test_trap_assert_stderr_unmatched("*prop4*");

    g_test_trap_assert_stderr("*Warning: \"-global nohotplug-type.prop5=105\" not used\n*");

    g_test_trap_assert_stderr("*Warning: \"-global nondevice-type.prop6=106\" not used\n*");

    g_test_trap_assert_stdout("");

}
