static void do_v7m_exception_exit(ARMCPU *cpu)

{

    CPUARMState *env = &cpu->env;

    uint32_t type;

    uint32_t xpsr;

    bool ufault = false;

    bool return_to_sp_process = false;

    bool return_to_handler = false;

    bool rettobase = false;



    /* We can only get here from an EXCP_EXCEPTION_EXIT, and

     * gen_bx_excret() enforces the architectural rule

     * that jumps to magic addresses don't have magic behaviour unless

     * we're in Handler mode (compare pseudocode BXWritePC()).

     */

    assert(arm_v7m_is_handler_mode(env));



    /* In the spec pseudocode ExceptionReturn() is called directly

     * from BXWritePC() and gets the full target PC value including

     * bit zero. In QEMU's implementation we treat it as a normal

     * jump-to-register (which is then caught later on), and so split

     * the target value up between env->regs[15] and env->thumb in

     * gen_bx(). Reconstitute it.

     */

    type = env->regs[15];

    if (env->thumb) {

        type |= 1;

    }



    qemu_log_mask(CPU_LOG_INT, "Exception return: magic PC %" PRIx32

                  " previous exception %d\n",

                  type, env->v7m.exception);



    if (extract32(type, 5, 23) != extract32(-1, 5, 23)) {

        qemu_log_mask(LOG_GUEST_ERROR, "M profile: zero high bits in exception "

                      "exit PC value 0x%" PRIx32 " are UNPREDICTABLE\n", type);

    }



    if (env->v7m.exception != ARMV7M_EXCP_NMI) {

        /* Auto-clear FAULTMASK on return from other than NMI.

         * If the security extension is implemented then this only

         * happens if the raw execution priority is >= 0; the

         * value of the ES bit in the exception return value indicates

         * which security state's faultmask to clear. (v8M ARM ARM R_KBNF.)

         */

        if (arm_feature(env, ARM_FEATURE_M_SECURITY)) {

            int es = type & 1;

            if (armv7m_nvic_raw_execution_priority(env->nvic) >= 0) {

                env->v7m.faultmask[es] = 0;

            }

        } else {

            env->v7m.faultmask[M_REG_NS] = 0;

        }

    }



    switch (armv7m_nvic_complete_irq(env->nvic, env->v7m.exception)) {

    case -1:

        /* attempt to exit an exception that isn't active */

        ufault = true;

        break;

    case 0:

        /* still an irq active now */

        break;

    case 1:

        /* we returned to base exception level, no nesting.

         * (In the pseudocode this is written using "NestedActivation != 1"

         * where we have 'rettobase == false'.)

         */

        rettobase = true;

        break;

    default:

        g_assert_not_reached();

    }



    switch (type & 0xf) {

    case 1: /* Return to Handler */

        return_to_handler = true;

        break;

    case 13: /* Return to Thread using Process stack */

        return_to_sp_process = true;

        /* fall through */

    case 9: /* Return to Thread using Main stack */

        if (!rettobase &&

            !(env->v7m.ccr[env->v7m.secure] & R_V7M_CCR_NONBASETHRDENA_MASK)) {

            ufault = true;

        }

        break;

    default:

        ufault = true;

    }



    if (ufault) {

        /* Bad exception return: instead of popping the exception

         * stack, directly take a usage fault on the current stack.

         */

        env->v7m.cfsr |= R_V7M_CFSR_INVPC_MASK;

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE);

        v7m_exception_taken(cpu, type | 0xf0000000);

        qemu_log_mask(CPU_LOG_INT, "...taking UsageFault on existing "

                      "stackframe: failed exception return integrity check\n");

        return;

    }



    /* Switch to the target stack.  */

    switch_v7m_sp(env, return_to_sp_process);

    /* Pop registers.  */

    env->regs[0] = v7m_pop(env);

    env->regs[1] = v7m_pop(env);

    env->regs[2] = v7m_pop(env);

    env->regs[3] = v7m_pop(env);

    env->regs[12] = v7m_pop(env);

    env->regs[14] = v7m_pop(env);

    env->regs[15] = v7m_pop(env);

    if (env->regs[15] & 1) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "M profile return from interrupt with misaligned "

                      "PC is UNPREDICTABLE\n");

        /* Actual hardware seems to ignore the lsbit, and there are several

         * RTOSes out there which incorrectly assume the r15 in the stack

         * frame should be a Thumb-style "lsbit indicates ARM/Thumb" value.

         */

        env->regs[15] &= ~1U;

    }

    xpsr = v7m_pop(env);

    xpsr_write(env, xpsr, ~XPSR_SPREALIGN);

    /* Undo stack alignment.  */

    if (xpsr & XPSR_SPREALIGN) {

        env->regs[13] |= 4;

    }



    /* The restored xPSR exception field will be zero if we're

     * resuming in Thread mode. If that doesn't match what the

     * exception return type specified then this is a UsageFault.

     */

    if (return_to_handler != arm_v7m_is_handler_mode(env)) {

        /* Take an INVPC UsageFault by pushing the stack again. */

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE);

        env->v7m.cfsr |= R_V7M_CFSR_INVPC_MASK;

        v7m_push_stack(cpu);

        v7m_exception_taken(cpu, type | 0xf0000000);

        qemu_log_mask(CPU_LOG_INT, "...taking UsageFault on new stackframe: "

                      "failed exception return integrity check\n");

        return;

    }



    /* Otherwise, we have a successful exception exit. */

    qemu_log_mask(CPU_LOG_INT, "...successful exception return\n");

}
