static void hpdmc_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                        unsigned size)

{

    MilkymistHpdmcState *s = opaque;



    trace_milkymist_hpdmc_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_SYSTEM:

    case R_BYPASS:

    case R_TIMING:

        s->regs[addr] = value;

        break;

    case R_IODELAY:

        /* ignore writes */

        break;



    default:

        error_report("milkymist_hpdmc: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}
