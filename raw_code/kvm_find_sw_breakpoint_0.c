struct kvm_sw_breakpoint *kvm_find_sw_breakpoint(CPUState *env,

                                                 target_ulong pc)

{

    struct kvm_sw_breakpoint *bp;



    TAILQ_FOREACH(bp, &env->kvm_state->kvm_sw_breakpoints, entry) {

        if (bp->pc == pc)

            return bp;

    }

    return NULL;

}
