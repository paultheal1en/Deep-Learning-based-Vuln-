static void gt_ctl_write(CPUARMState *env, const ARMCPRegInfo *ri,

                         uint64_t value)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    int timeridx = ri->crm & 1;

    uint32_t oldval = env->cp15.c14_timer[timeridx].ctl;



    env->cp15.c14_timer[timeridx].ctl = value & 3;

    if ((oldval ^ value) & 1) {

        /* Enable toggled */

        gt_recalc_timer(cpu, timeridx);

    } else if ((oldval & value) & 2) {

        /* IMASK toggled: don't need to recalculate,

         * just set the interrupt line based on ISTATUS

         */

        qemu_set_irq(cpu->gt_timer_outputs[timeridx],

                     (oldval & 4) && (value & 2));

    }

}
