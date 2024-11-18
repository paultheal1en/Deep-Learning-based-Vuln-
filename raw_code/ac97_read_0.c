static uint64_t ac97_read(void *opaque, target_phys_addr_t addr,

                          unsigned size)

{

    MilkymistAC97State *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_AC97_CTRL:

    case R_AC97_ADDR:

    case R_AC97_DATAOUT:

    case R_AC97_DATAIN:

    case R_D_CTRL:

    case R_D_ADDR:

    case R_D_REMAINING:

    case R_U_CTRL:

    case R_U_ADDR:

    case R_U_REMAINING:

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_ac97: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_ac97_memory_read(addr << 2, r);



    return r;

}
