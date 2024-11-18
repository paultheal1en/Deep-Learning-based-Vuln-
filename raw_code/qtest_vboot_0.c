QOSState *qtest_vboot(QOSOps *ops, const char *cmdline_fmt, va_list ap)

{

    char *cmdline;



    struct QOSState *qs = g_malloc(sizeof(QOSState));



    cmdline = g_strdup_vprintf(cmdline_fmt, ap);

    qs->qts = qtest_start(cmdline);

    qs->ops = ops;

    qtest_irq_intercept_in(global_qtest, "ioapic");

    if (ops && ops->init_allocator) {

        qs->alloc = ops->init_allocator(ALLOC_NO_FLAGS);

    }



    g_free(cmdline);

    return qs;

}
