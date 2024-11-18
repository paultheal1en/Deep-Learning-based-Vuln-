static void ahci_test_identify(AHCIQState *ahci)

{

    RegD2HFIS *d2h = g_malloc0(0x20);

    RegD2HFIS *pio = g_malloc0(0x20);

    RegH2DFIS fis;

    AHCICommandHeader cmd;

    PRD prd;

    uint32_t reg, data_ptr;

    uint16_t buff[256];

    unsigned i;

    int rc;

    uint8_t cx;

    uint64_t table;



    g_assert(ahci != NULL);



    /* We need to:

     * (1) Create a Command Table Buffer and update the Command List Slot #0

     *     to point to this buffer.

     * (2) Construct an FIS host-to-device command structure, and write it to

     *     the top of the command table buffer.

     * (3) Create a data buffer for the IDENTIFY response to be sent to

     * (4) Create a Physical Region Descriptor that points to the data buffer,

     *     and write it to the bottom (offset 0x80) of the command table.

     * (5) Now, PxCLB points to the command list, command 0 points to

     *     our table, and our table contains an FIS instruction and a

     *     PRD that points to our rx buffer.

     * (6) We inform the HBA via PxCI that there is a command ready in slot #0.

     */



    /* Pick the first implemented and running port */

    i = ahci_port_select(ahci);

    g_test_message("Selected port %u for test", i);



    /* Clear out the FIS Receive area and any pending interrupts. */

    ahci_port_clear(ahci, i);



    /* Create a Command Table buffer. 0x80 is the smallest with a PRDTL of 0. */

    /* We need at least one PRD, so round up to the nearest 0x80 multiple.    */

    table = ahci_alloc(ahci, CMD_TBL_SIZ(1));

    g_assert(table);

    ASSERT_BIT_CLEAR(table, 0x7F);



    /* Create a data buffer ... where we will dump the IDENTIFY data to. */

    data_ptr = ahci_alloc(ahci, 512);

    g_assert(data_ptr);



    /* pick a command slot (should be 0!) */

    cx = ahci_pick_cmd(ahci, i);



    /* Construct our Command Header (set_command_header handles endianness.) */

    memset(&cmd, 0x00, sizeof(cmd));

    cmd.flags = 5;      /* reg_h2d_fis is 5 double-words long */

    cmd.flags |= 0x400; /* clear PxTFD.STS.BSY when done */

    cmd.prdtl = 1;      /* One PRD table entry. */

    cmd.prdbc = 0;

    cmd.ctba = table;



    /* Construct our PRD, noting that DBC is 0-indexed. */

    prd.dba = cpu_to_le64(data_ptr);

    prd.res = 0;

    /* 511+1 bytes, request DPS interrupt */

    prd.dbc = cpu_to_le32(511 | 0x80000000);



    /* Construct our Command FIS, Based on http://wiki.osdev.org/AHCI */

    memset(&fis, 0x00, sizeof(fis));

    fis.fis_type = 0x27; /* Register Host-to-Device FIS */

    fis.command = 0xEC;  /* IDENTIFY */

    fis.device = 0;

    fis.flags = 0x80;    /* Indicate this is a command FIS */



    /* We've committed nothing yet, no interrupts should be posted yet. */

    g_assert_cmphex(ahci_px_rreg(ahci, i, AHCI_PX_IS), ==, 0);



    /* Commit the Command FIS to the Command Table */

    memwrite(table, &fis, sizeof(fis));



    /* Commit the PRD entry to the Command Table */

    memwrite(table + 0x80, &prd, sizeof(prd));



    /* Commit Command #cx, pointing to the Table, to the Command List Buffer. */

    ahci_set_command_header(ahci, i, cx, &cmd);



    /* Everything is in place, but we haven't given the go-ahead yet,

     * so we should find that there are no pending interrupts yet. */

    g_assert_cmphex(ahci_px_rreg(ahci, i, AHCI_PX_IS), ==, 0);



    /* Issue Command #cx via PxCI */

    ahci_px_wreg(ahci, i, AHCI_PX_CI, (1 << cx));

    while (BITSET(ahci_px_rreg(ahci, i, AHCI_PX_TFD), AHCI_PX_TFD_STS_BSY)) {

        usleep(50);

    }



    /* Check for expected interrupts */

    reg = ahci_px_rreg(ahci, i, AHCI_PX_IS);

    ASSERT_BIT_SET(reg, AHCI_PX_IS_DHRS);

    ASSERT_BIT_SET(reg, AHCI_PX_IS_PSS);

    /* BUG: we expect AHCI_PX_IS_DPS to be set. */

    ASSERT_BIT_CLEAR(reg, AHCI_PX_IS_DPS);



    /* Clear expected interrupts and assert all interrupts now cleared. */

    ahci_px_wreg(ahci, i, AHCI_PX_IS,

                 AHCI_PX_IS_DHRS | AHCI_PX_IS_PSS | AHCI_PX_IS_DPS);

    g_assert_cmphex(ahci_px_rreg(ahci, i, AHCI_PX_IS), ==, 0);



    /* Check for errors. */

    reg = ahci_px_rreg(ahci, i, AHCI_PX_SERR);

    g_assert_cmphex(reg, ==, 0);

    reg = ahci_px_rreg(ahci, i, AHCI_PX_TFD);

    ASSERT_BIT_CLEAR(reg, AHCI_PX_TFD_STS_ERR);

    ASSERT_BIT_CLEAR(reg, AHCI_PX_TFD_ERR);



    /* Investigate the CMD, assert that we read 512 bytes */

    ahci_get_command_header(ahci, i, cx, &cmd);

    g_assert_cmphex(512, ==, cmd.prdbc);



    /* Investigate FIS responses */

    memread(ahci->port[i].fb + 0x20, pio, 0x20);

    memread(ahci->port[i].fb + 0x40, d2h, 0x20);

    g_assert_cmphex(pio->fis_type, ==, 0x5f);

    g_assert_cmphex(d2h->fis_type, ==, 0x34);

    g_assert_cmphex(pio->flags, ==, d2h->flags);

    g_assert_cmphex(pio->status, ==, d2h->status);

    g_assert_cmphex(pio->error, ==, d2h->error);



    reg = ahci_px_rreg(ahci, i, AHCI_PX_TFD);

    g_assert_cmphex((reg & AHCI_PX_TFD_ERR), ==, pio->error);

    g_assert_cmphex((reg & AHCI_PX_TFD_STS), ==, pio->status);

    /* The PIO Setup FIS contains a "bytes read" field, which is a

     * 16-bit value. The Physical Region Descriptor Byte Count is

     * 32-bit, but for small transfers using one PRD, it should match. */

    g_assert_cmphex(le16_to_cpu(pio->res4), ==, cmd.prdbc);



    /* Last, but not least: Investigate the IDENTIFY response data. */

    memread(data_ptr, &buff, 512);



    /* Check serial number/version in the buffer */

    /* NB: IDENTIFY strings are packed in 16bit little endian chunks.

     * Since we copy byte-for-byte in ahci-test, on both LE and BE, we need to

     * unchunk this data. By contrast, ide-test copies 2 bytes at a time, and

     * as a consequence, only needs to unchunk the data on LE machines. */

    string_bswap16(&buff[10], 20);

    rc = memcmp(&buff[10], "testdisk            ", 20);

    g_assert_cmphex(rc, ==, 0);



    string_bswap16(&buff[23], 8);

    rc = memcmp(&buff[23], "version ", 8);

    g_assert_cmphex(rc, ==, 0);



    g_free(d2h);

    g_free(pio);

}
