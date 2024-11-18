static uint64_t mpc8544_guts_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    uint32_t value = 0;

    CPUPPCState *env = cpu_single_env;



    addr &= MPC8544_GUTS_MMIO_SIZE - 1;

    switch (addr) {

    case MPC8544_GUTS_ADDR_PVR:

        value = env->spr[SPR_PVR];

        break;

    case MPC8544_GUTS_ADDR_SVR:

        value = env->spr[SPR_E500_SVR];

        break;

    default:

        fprintf(stderr, "guts: Unknown register read: %x\n", (int)addr);

        break;

    }



    return value;

}
