static void pflash_cfi01_realize(DeviceState *dev, Error **errp)
{
    pflash_t *pfl = CFI_PFLASH01(dev);
    uint64_t total_len;
    int ret;
    uint64_t blocks_per_device, device_len;
    int num_devices;
    Error *local_err = NULL;
    total_len = pfl->sector_len * pfl->nb_blocs;
    /* These are only used to expose the parameters of each device
     * in the cfi_table[].
     */
    num_devices = pfl->device_width ? (pfl->bank_width / pfl->device_width) : 1;
    blocks_per_device = pfl->nb_blocs / num_devices;
    device_len = pfl->sector_len * blocks_per_device;
    /* XXX: to be fixed */
#if 0
    if (total_len != (8 * 1024 * 1024) && total_len != (16 * 1024 * 1024) &&
        total_len != (32 * 1024 * 1024) && total_len != (64 * 1024 * 1024))
        return NULL;
#endif
    memory_region_init_rom_device(
        &pfl->mem, OBJECT(dev),
        &pflash_cfi01_ops,
        pfl,
        pfl->name, total_len, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
    vmstate_register_ram(&pfl->mem, DEVICE(pfl));
    pfl->storage = memory_region_get_ram_ptr(&pfl->mem);
    sysbus_init_mmio(SYS_BUS_DEVICE(dev), &pfl->mem);
    if (pfl->blk) {
        /* read the initial flash content */
        ret = blk_pread(pfl->blk, 0, pfl->storage, total_len);
        if (ret < 0) {
            vmstate_unregister_ram(&pfl->mem, DEVICE(pfl));
            error_setg(errp, "failed to read the initial flash content");
    if (pfl->blk) {
        pfl->ro = blk_is_read_only(pfl->blk);
    } else {
        pfl->ro = 0;
    /* Default to devices being used at their maximum device width. This was
     * assumed before the device_width support was added.
     */
    if (!pfl->max_device_width) {
        pfl->max_device_width = pfl->device_width;
    pfl->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, pflash_timer, pfl);
    pfl->wcycle = 0;
    pfl->cmd = 0;
    pfl->status = 0;
    /* Hardcoded CFI table */
    pfl->cfi_len = 0x52;
    /* Standard "QRY" string */
    pfl->cfi_table[0x10] = 'Q';
    pfl->cfi_table[0x11] = 'R';
    pfl->cfi_table[0x12] = 'Y';
    /* Command set (Intel) */
    pfl->cfi_table[0x13] = 0x01;
    pfl->cfi_table[0x14] = 0x00;
    /* Primary extended table address (none) */
    pfl->cfi_table[0x15] = 0x31;
    pfl->cfi_table[0x16] = 0x00;
    /* Alternate command set (none) */
    pfl->cfi_table[0x17] = 0x00;
    pfl->cfi_table[0x18] = 0x00;
    /* Alternate extended table (none) */
    pfl->cfi_table[0x19] = 0x00;
    pfl->cfi_table[0x1A] = 0x00;
    /* Vcc min */
    pfl->cfi_table[0x1B] = 0x45;
    /* Vcc max */
    pfl->cfi_table[0x1C] = 0x55;
    /* Vpp min (no Vpp pin) */
    pfl->cfi_table[0x1D] = 0x00;
    /* Vpp max (no Vpp pin) */
    pfl->cfi_table[0x1E] = 0x00;
    /* Reserved */
    pfl->cfi_table[0x1F] = 0x07;
    /* Timeout for min size buffer write */
    pfl->cfi_table[0x20] = 0x07;
    /* Typical timeout for block erase */
    pfl->cfi_table[0x21] = 0x0a;
    /* Typical timeout for full chip erase (4096 ms) */
    pfl->cfi_table[0x22] = 0x00;
    /* Reserved */
    pfl->cfi_table[0x23] = 0x04;
    /* Max timeout for buffer write */
    pfl->cfi_table[0x24] = 0x04;
    /* Max timeout for block erase */
    pfl->cfi_table[0x25] = 0x04;
    /* Max timeout for chip erase */
    pfl->cfi_table[0x26] = 0x00;
    /* Device size */
    pfl->cfi_table[0x27] = ctz32(device_len); /* + 1; */
    /* Flash device interface (8 & 16 bits) */
    pfl->cfi_table[0x28] = 0x02;
    pfl->cfi_table[0x29] = 0x00;
    /* Max number of bytes in multi-bytes write */
    if (pfl->bank_width == 1) {
        pfl->cfi_table[0x2A] = 0x08;
    } else {
        pfl->cfi_table[0x2A] = 0x0B;
    pfl->writeblock_size = 1 << pfl->cfi_table[0x2A];
    pfl->cfi_table[0x2B] = 0x00;
    /* Number of erase block regions (uniform) */
    pfl->cfi_table[0x2C] = 0x01;
    /* Erase block region 1 */
    pfl->cfi_table[0x2D] = blocks_per_device - 1;
    pfl->cfi_table[0x2E] = (blocks_per_device - 1) >> 8;
    pfl->cfi_table[0x2F] = pfl->sector_len >> 8;
    pfl->cfi_table[0x30] = pfl->sector_len >> 16;
    /* Extended */
    pfl->cfi_table[0x31] = 'P';
    pfl->cfi_table[0x32] = 'R';
    pfl->cfi_table[0x33] = 'I';
    pfl->cfi_table[0x34] = '1';
    pfl->cfi_table[0x35] = '0';
    pfl->cfi_table[0x36] = 0x00;
    pfl->cfi_table[0x37] = 0x00;
    pfl->cfi_table[0x38] = 0x00;
    pfl->cfi_table[0x39] = 0x00;
    pfl->cfi_table[0x3a] = 0x00;
    pfl->cfi_table[0x3b] = 0x00;
    pfl->cfi_table[0x3c] = 0x00;
    pfl->cfi_table[0x3f] = 0x01; /* Number of protection fields */