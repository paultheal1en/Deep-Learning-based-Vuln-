static void tmu2_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                       unsigned size)

{

    MilkymistTMU2State *s = opaque;



    trace_milkymist_tmu2_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_CTL:

        s->regs[addr] = value;

        if (value & CTL_START_BUSY) {

            tmu2_start(s);

        }

        break;

    case R_BRIGHTNESS:

    case R_HMESHLAST:

    case R_VMESHLAST:

    case R_CHROMAKEY:

    case R_VERTICESADDR:

    case R_TEXFBUF:

    case R_TEXHRES:

    case R_TEXVRES:

    case R_TEXHMASK:

    case R_TEXVMASK:

    case R_DSTFBUF:

    case R_DSTHRES:

    case R_DSTVRES:

    case R_DSTHOFFSET:

    case R_DSTVOFFSET:

    case R_DSTSQUAREW:

    case R_DSTSQUAREH:

    case R_ALPHA:

        s->regs[addr] = value;

        break;



    default:

        error_report("milkymist_tmu2: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    tmu2_check_registers(s);

}
