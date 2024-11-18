static uint64_t ich_elrsr_read(CPUARMState *env, const ARMCPRegInfo *ri)

{

    GICv3CPUState *cs = icc_cs_from_env(env);

    uint64_t value = 0;

    int i;



    for (i = 0; i < cs->num_list_regs; i++) {

        uint64_t lr = cs->ich_lr_el2[i];



        if ((lr & ICH_LR_EL2_STATE_MASK) == 0 &&

            ((lr & ICH_LR_EL2_HW) == 1 || (lr & ICH_LR_EL2_EOI) == 0)) {

            value |= (1 << i);

        }

    }



    trace_gicv3_ich_elrsr_read(gicv3_redist_affid(cs), value);

    return value;

}
