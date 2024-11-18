static void cmd646_cmd_write(void *opaque, target_phys_addr_t addr,

                             uint64_t data, unsigned size)

{

    CMD646BAR *cmd646bar = opaque;



    if (addr != 2 || size != 1) {

        return;

    }

    ide_cmd_write(cmd646bar->bus, addr + 2, data);

}
