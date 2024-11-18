static void vmgenid_set_guid_auto_test(void)

{

    const char *cmd;

    QemuUUID measured;



    cmd = "-machine accel=tcg -device vmgenid,id=testvgid," "guid=auto";

    qtest_start(cmd);



    read_guid_from_memory(&measured);



    /* Just check that the GUID is non-null */

    g_assert(!qemu_uuid_is_null(&measured));



    qtest_quit(global_qtest);

}
