static uint64_t vgafb_read(void *opaque, target_phys_addr_t addr,

                           unsigned size)

{

    MilkymistVgafbState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_CTRL:

    case R_HRES:

    case R_HSYNC_START:

    case R_HSYNC_END:

    case R_HSCAN:

    case R_VRES:

    case R_VSYNC_START:

    case R_VSYNC_END:

    case R_VSCAN:

    case R_BASEADDRESS:

    case R_BURST_COUNT:

    case R_DDC:

    case R_SOURCE_CLOCK:

        r = s->regs[addr];

    break;

    case R_BASEADDRESS_ACT:

        r = s->regs[R_BASEADDRESS];

    break;



    default:

        error_report("milkymist_vgafb: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_vgafb_memory_read(addr << 2, r);



    return r;

}
