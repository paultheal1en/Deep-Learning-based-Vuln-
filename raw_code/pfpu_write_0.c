static void pfpu_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                       unsigned size)

{

    MilkymistPFPUState *s = opaque;



    trace_milkymist_pfpu_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_CTL:

        if (value & CTL_START_BUSY) {

            pfpu_start(s);

        }

        break;

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

        s->regs[addr] = value;

        break;

    case GPR_BEGIN ...  GPR_END:

        s->gp_regs[addr - GPR_BEGIN] = value;

        break;

    case MICROCODE_BEGIN ...  MICROCODE_END:

        s->microcode[get_microcode_address(s, addr)] = value;

        break;



    default:

        error_report("milkymist_pfpu: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}
