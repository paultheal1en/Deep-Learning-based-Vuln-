static void hb_regs_write(void *opaque, hwaddr offset,

                          uint64_t value, unsigned size)

{

    uint32_t *regs = opaque;



    if (offset == 0xf00) {

        if (value == 1 || value == 2) {

            qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET);

        } else if (value == 3) {

            qemu_system_shutdown_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN);

        }

    }



    regs[offset/4] = value;

}
