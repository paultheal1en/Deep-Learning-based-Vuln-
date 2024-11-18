static int bad_mode_switch(CPUARMState *env, int mode)

{

    /* Return true if it is not valid for us to switch to

     * this CPU mode (ie all the UNPREDICTABLE cases in

     * the ARM ARM CPSRWriteByInstr pseudocode).

     */

    switch (mode) {

    case ARM_CPU_MODE_USR:

    case ARM_CPU_MODE_SYS:

    case ARM_CPU_MODE_SVC:

    case ARM_CPU_MODE_ABT:

    case ARM_CPU_MODE_UND:

    case ARM_CPU_MODE_IRQ:

    case ARM_CPU_MODE_FIQ:

        /* Note that we don't implement the IMPDEF NSACR.RFR which in v7

         * allows FIQ mode to be Secure-only. (In v8 this doesn't exist.)

         */

        return 0;

    case ARM_CPU_MODE_HYP:

        return !arm_feature(env, ARM_FEATURE_EL2)

            || arm_current_el(env) < 2 || arm_is_secure(env);

    case ARM_CPU_MODE_MON:

        return !arm_is_secure(env);

    default:

        return 1;

    }

}
