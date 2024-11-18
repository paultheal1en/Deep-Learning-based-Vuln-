static QOSState *qvirtio_9p_start(void)

{

    const char *cmd = "-fsdev local,id=fsdev0,security_model=none,path=%s "

                      "-device virtio-9p-pci,fsdev=fsdev0,mount_tag=%s";



    test_share = g_strdup("/tmp/qtest.XXXXXX");

    g_assert_nonnull(mkdtemp(test_share));



    return qtest_pc_boot(cmd, test_share, mount_tag);

}
