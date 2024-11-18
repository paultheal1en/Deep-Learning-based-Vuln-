static int send_dma_request(int cmd, uint64_t sector, int nb_sectors,
                            PrdtEntry *prdt, int prdt_entries)
{
    QPCIDevice *dev;
    uint16_t bmdma_base;
    uintptr_t guest_prdt;
    size_t len;
    bool from_dev;
    uint8_t status;
    int flags;
    dev = get_pci_device(&bmdma_base);
    flags = cmd & ~0xff;
    cmd &= 0xff;
    switch (cmd) {
    case CMD_READ_DMA:
        from_dev = true;
        break;
    case CMD_WRITE_DMA:
        from_dev = false;
        break;
    default:
        g_assert_not_reached();
    /* Select device 0 */
    outb(IDE_BASE + reg_device, 0 | LBA);
    /* Stop any running transfer, clear any pending interrupt */
    outb(bmdma_base + bmreg_cmd, 0);
    outb(bmdma_base + bmreg_status, BM_STS_INTR);
    /* Setup PRDT */
    len = sizeof(*prdt) * prdt_entries;
    guest_prdt = guest_alloc(guest_malloc, len);
    memwrite(guest_prdt, prdt, len);
    outl(bmdma_base + bmreg_prdt, guest_prdt);
    /* ATA DMA command */
    outb(IDE_BASE + reg_nsectors, nb_sectors);
    outb(IDE_BASE + reg_lba_low,    sector & 0xff);
    outb(IDE_BASE + reg_lba_middle, (sector >> 8) & 0xff);
    outb(IDE_BASE + reg_lba_high,   (sector >> 16) & 0xff);
    outb(IDE_BASE + reg_command, cmd);
    /* Start DMA transfer */
    outb(bmdma_base + bmreg_cmd, BM_CMD_START | (from_dev ? BM_CMD_WRITE : 0));
    if (flags & CMDF_ABORT) {
        outb(bmdma_base + bmreg_cmd, 0);
    /* Wait for the DMA transfer to complete */
    do {
        status = inb(bmdma_base + bmreg_status);
    } while ((status & (BM_STS_ACTIVE | BM_STS_INTR)) == BM_STS_ACTIVE);
    g_assert_cmpint(get_irq(IDE_PRIMARY_IRQ), ==, !!(status & BM_STS_INTR));
    /* Check IDE status code */
    assert_bit_set(inb(IDE_BASE + reg_status), DRDY);
    assert_bit_clear(inb(IDE_BASE + reg_status), BSY | DRQ);
    /* Reading the status register clears the IRQ */
    g_assert(!get_irq(IDE_PRIMARY_IRQ));
    /* Stop DMA transfer if still active */
    if (status & BM_STS_ACTIVE) {
        outb(bmdma_base + bmreg_cmd, 0);
    free_pci_device(dev);
    return status;