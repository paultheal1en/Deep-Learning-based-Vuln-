static uint64_t tmu2_read(void *opaque, target_phys_addr_t addr,

                          unsigned size)

{

    MilkymistTMU2State *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_CTL:

    case R_HMESHLAST:

    case R_VMESHLAST:

    case R_BRIGHTNESS:

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

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_tmu2: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_tmu2_memory_read(addr << 2, r);



    return r;

}
