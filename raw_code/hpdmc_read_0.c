static uint64_t hpdmc_read(void *opaque, target_phys_addr_t addr,

                           unsigned size)

{

    MilkymistHpdmcState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_SYSTEM:

    case R_BYPASS:

    case R_TIMING:

    case R_IODELAY:

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_hpdmc: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_hpdmc_memory_read(addr << 2, r);



    return r;

}
