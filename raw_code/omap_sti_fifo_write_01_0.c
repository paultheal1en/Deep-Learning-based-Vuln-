static void omap_sti_fifo_write(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    struct omap_sti_s *s = (struct omap_sti_s *) opaque;

    int offset = addr - s->channel_base;

    int ch = offset >> 6;

    uint8_t byte = value;



    if (ch == STI_TRACE_CONTROL_CHANNEL) {

        /* Flush channel <i>value</i>.  */

        qemu_chr_write(s->chr, "\r", 1);

    } else if (ch == STI_TRACE_CONSOLE_CHANNEL || 1) {

        if (value == 0xc0 || value == 0xc3) {

            /* Open channel <i>ch</i>.  */

        } else if (value == 0x00)

            qemu_chr_write(s->chr, "\n", 1);

        else

            qemu_chr_write(s->chr, &byte, 1);

    }

}
