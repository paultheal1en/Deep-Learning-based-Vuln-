static inline int fp_reg_hi_offset(int regno)

{

    return offsetof(CPUARMState, vfp.regs[regno * 2 + 1]);

}
