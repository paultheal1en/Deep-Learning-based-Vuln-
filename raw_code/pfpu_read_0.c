static uint64_t pfpu_read(void *opaque, target_phys_addr_t addr,

                          unsigned size)

{

    MilkymistPFPUState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_CTL:

    case R_MESHBASE:

    case R_HMESHLAST:

    case R_VMESHLAST:

    case R_CODEPAGE:

    case R_VERTICES:

    case R_COLLISIONS:

    case R_STRAYWRITES:

    case R_LASTDMA:

    case R_PC:

    case R_DREGBASE:

    case R_CODEBASE:

        r = s->regs[addr];

        break;

    case GPR_BEGIN ... GPR_END:

        r = s->gp_regs[addr - GPR_BEGIN];

        break;

    case MICROCODE_BEGIN ...  MICROCODE_END:

        r = s->microcode[get_microcode_address(s, addr)];

        break;



    default:

        error_report("milkymist_pfpu: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_pfpu_memory_read(addr << 2, r);



    return r;

}
