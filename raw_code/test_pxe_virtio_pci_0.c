static void test_pxe_virtio_pci(void)

{

    test_pxe_one("-device virtio-net-pci,netdev=" NETNAME, false);

}
