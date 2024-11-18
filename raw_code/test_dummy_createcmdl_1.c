static void test_dummy_createcmdl(void)
{
    QemuOpts *opts;
    DummyObject *dobj;
    Error *err = NULL;
    const char *params = TYPE_DUMMY \
                         ",id=dev0," \
                         "bv=yes,sv=Hiss hiss hiss,av=platypus";
    qemu_add_opts(&qemu_object_opts);
    opts = qemu_opts_parse(&qemu_object_opts, params, true, &err);
    g_assert(err == NULL);
    g_assert(opts);
    dobj = DUMMY_OBJECT(user_creatable_add_opts(opts, &err));
    g_assert(err == NULL);
    g_assert(dobj);
    g_assert_cmpstr(dobj->sv, ==, "Hiss hiss hiss");
    g_assert(dobj->bv == true);
    g_assert(dobj->av == DUMMY_PLATYPUS);
    user_creatable_del("dev0", &err);
    g_assert(err == NULL);
    error_free(err);
    /*
     * cmdline-parsing via qemu_opts_parse() results in a QemuOpts entry
     * corresponding to the Object's ID to be added to the QemuOptsList
     * for objects. To avoid having this entry conflict with future
     * Objects using the same ID (which can happen in cases where
     * qemu_opts_parse() is used to parse the object params, such as
     * with hmp_object_add() at the time of this comment), we need to
     * check for this in user_creatable_del() and remove the QemuOpts if
     * it is present.
     *
     * The below check ensures this works as expected.
     */
    g_assert_null(qemu_opts_find(&qemu_object_opts, "dev0"));
}