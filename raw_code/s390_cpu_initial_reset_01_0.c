static void s390_cpu_initial_reset(CPUState *s)

{

    S390CPU *cpu = S390_CPU(s);

    CPUS390XState *env = &cpu->env;

    int i;



    s390_cpu_reset(s);

    /* initial reset does not clear everything! */

    memset(&env->start_initial_reset_fields, 0,

        offsetof(CPUS390XState, end_reset_fields) -

        offsetof(CPUS390XState, start_initial_reset_fields));



    /* architectured initial values for CR 0 and 14 */

    env->cregs[0] = CR0_RESET;

    env->cregs[14] = CR14_RESET;



    /* architectured initial value for Breaking-Event-Address register */

    env->gbea = 1;



    env->pfault_token = -1UL;

    env->ext_index = -1;

    for (i = 0; i < ARRAY_SIZE(env->io_index); i++) {

        env->io_index[i] = -1;

    }

    env->mchk_index = -1;



    /* tininess for underflow is detected before rounding */

    set_float_detect_tininess(float_tininess_before_rounding,

                              &env->fpu_status);



    /* Reset state inside the kernel that we cannot access yet from QEMU. */

    if (kvm_enabled()) {

        kvm_s390_reset_vcpu(cpu);

    }

}
