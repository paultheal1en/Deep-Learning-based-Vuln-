static uint64_t cmd646_cmd_read(void *opaque, target_phys_addr_t addr,

                                unsigned size)

{

    CMD646BAR *cmd646bar = opaque;



    if (addr != 2 || size != 1) {

        return ((uint64_t)1 << (size * 8)) - 1;

    }

    return ide_status_read(cmd646bar->bus, addr + 2);

}
