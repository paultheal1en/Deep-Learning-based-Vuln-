static void pmsav7_write(CPUARMState *env, const ARMCPRegInfo *ri,

                         uint64_t value)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    uint32_t *u32p = *(uint32_t **)raw_ptr(env, ri);



    if (!u32p) {

        return;

    }



    u32p += env->pmsav7.rnr;

    tlb_flush(CPU(cpu)); /* Mappings may have changed - purge! */

    *u32p = value;

}
