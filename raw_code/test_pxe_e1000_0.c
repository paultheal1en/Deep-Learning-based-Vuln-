static void test_pxe_e1000(void)

{

    test_pxe_one("-device e1000,netdev=" NETNAME, false);

}
