pflash_t *pflash_cfi02_register(target_phys_addr_t base, ram_addr_t off,

                                BlockDriverState *bs, uint32_t sector_len,

                                int nb_blocs, int nb_mappings, int width,

                                uint16_t id0, uint16_t id1,

                                uint16_t id2, uint16_t id3,

                                uint16_t unlock_addr0, uint16_t unlock_addr1)

{

    pflash_t *pfl;

    int32_t chip_len;



    chip_len = sector_len * nb_blocs;

    /* XXX: to be fixed */

#if 0

    if (total_len != (8 * 1024 * 1024) && total_len != (16 * 1024 * 1024) &&

        total_len != (32 * 1024 * 1024) && total_len != (64 * 1024 * 1024))

        return NULL;

#endif

    pfl = qemu_mallocz(sizeof(pflash_t));

    /* FIXME: Allocate ram ourselves.  */

    pfl->storage = qemu_get_ram_ptr(off);

    pfl->fl_mem = cpu_register_io_memory(pflash_read_ops, pflash_write_ops,

                                         pfl);

    pfl->off = off;

    pfl->base = base;

    pfl->chip_len = chip_len;

    pfl->mappings = nb_mappings;

    pflash_register_memory(pfl, 1);

    pfl->bs = bs;

    if (pfl->bs) {

        /* read the initial flash content */

        bdrv_read(pfl->bs, 0, pfl->storage, chip_len >> 9);

    }

#if 0 /* XXX: there should be a bit to set up read-only,

       *      the same way the hardware does (with WP pin).

       */

    pfl->ro = 1;

#else

    pfl->ro = 0;

#endif

    pfl->timer = qemu_new_timer(vm_clock, pflash_timer, pfl);

    pfl->sector_len = sector_len;

    pfl->width = width;

    pfl->wcycle = 0;

    pfl->cmd = 0;

    pfl->status = 0;

    pfl->ident[0] = id0;

    pfl->ident[1] = id1;

    pfl->ident[2] = id2;

    pfl->ident[3] = id3;

    pfl->unlock_addr[0] = unlock_addr0;

    pfl->unlock_addr[1] = unlock_addr1;

    /* Hardcoded CFI table (mostly from SG29 Spansion flash) */

    pfl->cfi_len = 0x52;

    /* Standard "QRY" string */

    pfl->cfi_table[0x10] = 'Q';

    pfl->cfi_table[0x11] = 'R';

    pfl->cfi_table[0x12] = 'Y';

    /* Command set (AMD/Fujitsu) */

    pfl->cfi_table[0x13] = 0x02;

    pfl->cfi_table[0x14] = 0x00;

    /* Primary extended table address */

    pfl->cfi_table[0x15] = 0x31;

    pfl->cfi_table[0x16] = 0x00;

    /* Alternate command set (none) */

    pfl->cfi_table[0x17] = 0x00;

    pfl->cfi_table[0x18] = 0x00;

    /* Alternate extended table (none) */

    pfl->cfi_table[0x19] = 0x00;

    pfl->cfi_table[0x1A] = 0x00;

    /* Vcc min */

    pfl->cfi_table[0x1B] = 0x27;

    /* Vcc max */

    pfl->cfi_table[0x1C] = 0x36;

    /* Vpp min (no Vpp pin) */

    pfl->cfi_table[0x1D] = 0x00;

    /* Vpp max (no Vpp pin) */

    pfl->cfi_table[0x1E] = 0x00;

    /* Reserved */

    pfl->cfi_table[0x1F] = 0x07;

    /* Timeout for min size buffer write (NA) */

    pfl->cfi_table[0x20] = 0x00;

    /* Typical timeout for block erase (512 ms) */

    pfl->cfi_table[0x21] = 0x09;

    /* Typical timeout for full chip erase (4096 ms) */

    pfl->cfi_table[0x22] = 0x0C;

    /* Reserved */

    pfl->cfi_table[0x23] = 0x01;

    /* Max timeout for buffer write (NA) */

    pfl->cfi_table[0x24] = 0x00;

    /* Max timeout for block erase */

    pfl->cfi_table[0x25] = 0x0A;

    /* Max timeout for chip erase */

    pfl->cfi_table[0x26] = 0x0D;

    /* Device size */

    pfl->cfi_table[0x27] = ctz32(chip_len);

    /* Flash device interface (8 & 16 bits) */

    pfl->cfi_table[0x28] = 0x02;

    pfl->cfi_table[0x29] = 0x00;

    /* Max number of bytes in multi-bytes write */

    /* XXX: disable buffered write as it's not supported */

    //    pfl->cfi_table[0x2A] = 0x05;

    pfl->cfi_table[0x2A] = 0x00;

    pfl->cfi_table[0x2B] = 0x00;

    /* Number of erase block regions (uniform) */

    pfl->cfi_table[0x2C] = 0x01;

    /* Erase block region 1 */

    pfl->cfi_table[0x2D] = nb_blocs - 1;

    pfl->cfi_table[0x2E] = (nb_blocs - 1) >> 8;

    pfl->cfi_table[0x2F] = sector_len >> 8;

    pfl->cfi_table[0x30] = sector_len >> 16;



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



    return pfl;

}
