static void ahci_test_identify(AHCIQState *ahci)
{
    uint16_t buff[256];
    unsigned px;
    int rc;
    uint16_t sect_size;
    const size_t buffsize = 512;
    g_assert(ahci != NULL);
    /**
     * This serves as a bit of a tutorial on AHCI device programming:
     *
     * (1) Create a data buffer for the IDENTIFY response to be sent to
     * (2) Create a Command Table buffer, where we will store the
     *     command and PRDT (Physical Region Descriptor Table)
     * (3) Construct an FIS host-to-device command structure, and write it to
     *     the top of the Command Table buffer.
     * (4) Create one or more Physical Region Descriptors (PRDs) that describe
     *     a location in memory where data may be stored/retrieved.
     * (5) Write these PRDTs to the bottom (offset 0x80) of the Command Table.
     * (6) Each AHCI port has up to 32 command slots. Each slot contains a
     *     header that points to a Command Table buffer. Pick an unused slot
     *     and update it to point to the Command Table we have built.
     * (7) Now: Command #n points to our Command Table, and our Command Table
     *     contains the FIS (that describes our command) and the PRDTL, which
     *     describes our buffer.
     * (8) We inform the HBA via PxCI (Command Issue) that the command in slot
     *     #n is ready for processing.
     */
    /* Pick the first implemented and running port */
    px = ahci_port_select(ahci);
    g_test_message("Selected port %u for test", px);
    /* Clear out the FIS Receive area and any pending interrupts. */
    ahci_port_clear(ahci, px);
    /* "Read" 512 bytes using CMD_IDENTIFY into the host buffer. */
    ahci_io(ahci, px, CMD_IDENTIFY, &buff, buffsize);
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
}