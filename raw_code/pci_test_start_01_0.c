static QPCIBus *pci_test_start(int socket)

{

    char *cmdline;



    cmdline = g_strdup_printf("-netdev socket,fd=%d,id=hs0 -device "

                              "virtio-net-pci,netdev=hs0", socket);

    qtest_start(cmdline);

    g_free(cmdline);



    return qpci_init_pc(NULL);

}
