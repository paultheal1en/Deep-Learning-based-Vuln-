static void add_cpreg_to_list(gpointer key, gpointer opaque)

{

    ARMCPU *cpu = opaque;

    uint64_t regidx;

    const ARMCPRegInfo *ri;



    regidx = *(uint32_t *)key;

    ri = get_arm_cp_reginfo(cpu->cp_regs, regidx);



    if (!(ri->type & ARM_CP_NO_MIGRATE)) {

        cpu->cpreg_indexes[cpu->cpreg_array_len] = cpreg_to_kvm_id(regidx);

        /* The value array need not be initialized at this point */

        cpu->cpreg_array_len++;

    }

}
