void HELPER(set_cp_reg64)(CPUARMState *env, void *rip, uint64_t value)

{

    const ARMCPRegInfo *ri = rip;



    ri->writefn(env, ri, value);

}
