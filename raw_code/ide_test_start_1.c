static void ide_test_start(const char *cmdline_fmt, ...)
{
    va_list ap;
    char *cmdline;
    va_start(ap, cmdline_fmt);
    cmdline = g_strdup_vprintf(cmdline_fmt, ap);
    va_end(ap);
    qtest_start(cmdline);
    qtest_irq_intercept_in(global_qtest, "ioapic");
    guest_malloc = pc_alloc_init();
}