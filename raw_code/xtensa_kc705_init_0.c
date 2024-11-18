static void xtensa_kc705_init(MachineState *machine)

{

    static const LxBoardDesc kc705_board = {

        .flash_base = 0xf0000000,

        .flash_size = 0x08000000,

        .flash_boot_base = 0x06000000,

        .flash_sector_size = 0x20000,

        .sram_size = 0x2000000,

    };

    lx_init(&kc705_board, machine);

}
