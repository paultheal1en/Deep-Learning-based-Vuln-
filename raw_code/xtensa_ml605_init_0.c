static void xtensa_ml605_init(MachineState *machine)

{

    static const LxBoardDesc ml605_board = {

        .flash_base = 0xf8000000,

        .flash_size = 0x01000000,

        .flash_sector_size = 0x20000,

        .sram_size = 0x2000000,

    };

    lx_init(&ml605_board, machine);

}
