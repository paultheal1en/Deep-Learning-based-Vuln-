static void test_i440fx_pam(gconstpointer opaque)
{
    const TestData *s = opaque;
    QPCIBus *bus;
    QPCIDevice *dev;
    int i;
    static struct {
        uint32_t start;
        uint32_t end;
    } pam_area[] = {
        { 0, 0 },             /* Reserved */
        { 0xF0000, 0xFFFFF }, /* BIOS Area */
        { 0xC0000, 0xC3FFF }, /* Option ROM */
        { 0xC4000, 0xC7FFF }, /* Option ROM */
        { 0xC8000, 0xCBFFF }, /* Option ROM */
        { 0xCC000, 0xCFFFF }, /* Option ROM */
        { 0xD0000, 0xD3FFF }, /* Option ROM */
        { 0xD4000, 0xD7FFF }, /* Option ROM */
        { 0xD8000, 0xDBFFF }, /* Option ROM */
        { 0xDC000, 0xDFFFF }, /* Option ROM */
        { 0xE0000, 0xE3FFF }, /* BIOS Extension */
        { 0xE4000, 0xE7FFF }, /* BIOS Extension */
        { 0xE8000, 0xEBFFF }, /* BIOS Extension */
        { 0xEC000, 0xEFFFF }, /* BIOS Extension */
    };
    bus = test_start_get_bus(s);
    dev = qpci_device_find(bus, QPCI_DEVFN(0, 0));
    g_assert(dev != NULL);
    for (i = 0; i < ARRAY_SIZE(pam_area); i++) {
        if (pam_area[i].start == pam_area[i].end) {
            continue;
        }
        g_test_message("Checking area 0x%05x..0x%05x",
                       pam_area[i].start, pam_area[i].end);
        /* Switch to RE for the area */
        pam_set(dev, i, PAM_RE);
        /* Verify the RAM is all zeros */
        g_assert(verify_area(pam_area[i].start, pam_area[i].end, 0));
        /* Switch to WE for the area */
        pam_set(dev, i, PAM_RE | PAM_WE);
        /* Write out a non-zero mask to the full area */
        write_area(pam_area[i].start, pam_area[i].end, 0x42);
#ifndef BROKEN
        /* QEMU only supports a limited form of PAM */
        /* Switch to !RE for the area */
        pam_set(dev, i, PAM_WE);
        /* Verify the area is not our mask */
        g_assert(!verify_area(pam_area[i].start, pam_area[i].end, 0x42));
#endif
        /* Verify the area is our new mask */
        g_assert(verify_area(pam_area[i].start, pam_area[i].end, 0x42));
        /* Write out a new mask */
        write_area(pam_area[i].start, pam_area[i].end, 0x82);
#ifndef BROKEN
        /* QEMU only supports a limited form of PAM */
        /* Verify the area is not our mask */
        g_assert(!verify_area(pam_area[i].start, pam_area[i].end, 0x82));
        /* Switch to RE for the area */
        pam_set(dev, i, PAM_RE | PAM_WE);
#endif
        /* Verify the area is our new mask */
        g_assert(verify_area(pam_area[i].start, pam_area[i].end, 0x82));
        /* Reset area */
        pam_set(dev, i, 0);
        /* Verify the area is not our new mask */
        g_assert(!verify_area(pam_area[i].start, pam_area[i].end, 0x82));
    }
    qtest_end();
}