int kvm_sw_breakpoints_active(CPUState *env)

{

    return !TAILQ_EMPTY(&env->kvm_state->kvm_sw_breakpoints);

}
