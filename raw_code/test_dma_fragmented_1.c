static void test_dma_fragmented(void)

{

    AHCIQState *ahci;

    AHCICommand *cmd;

    uint8_t px;

    size_t bufsize = 4096;

    unsigned char *tx = g_malloc(bufsize);

    unsigned char *rx = g_malloc0(bufsize);

    uint64_t ptr;



    ahci = ahci_boot_and_enable(NULL);

    px = ahci_port_select(ahci);

    ahci_port_clear(ahci, px);



    /* create pattern */

    generate_pattern(tx, bufsize, AHCI_SECTOR_SIZE);



    /* Create a DMA buffer in guest memory, and write our pattern to it. */

    ptr = guest_alloc(ahci->parent->alloc, bufsize);

    g_assert(ptr);

    bufwrite(ptr, tx, bufsize);



    cmd = ahci_command_create(CMD_WRITE_DMA);

    ahci_command_adjust(cmd, 0, ptr, bufsize, 32);

    ahci_command_commit(ahci, cmd, px);

    ahci_command_issue(ahci, cmd);

    ahci_command_verify(ahci, cmd);

    g_free(cmd);



    cmd = ahci_command_create(CMD_READ_DMA);

    ahci_command_adjust(cmd, 0, ptr, bufsize, 32);

    ahci_command_commit(ahci, cmd, px);

    ahci_command_issue(ahci, cmd);

    ahci_command_verify(ahci, cmd);

    g_free(cmd);



    /* Read back the guest's receive buffer into local memory */

    bufread(ptr, rx, bufsize);

    guest_free(ahci->parent->alloc, ptr);



    g_assert_cmphex(memcmp(tx, rx, bufsize), ==, 0);



    ahci_shutdown(ahci);



    g_free(rx);

    g_free(tx);

}
