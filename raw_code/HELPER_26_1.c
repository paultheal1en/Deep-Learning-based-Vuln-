void HELPER(access_check_cp_reg)(CPUARMState *env, void *rip, uint32_t syndrome)
{
    const ARMCPRegInfo *ri = rip;
    switch (ri->accessfn(env, ri)) {
    case CP_ACCESS_OK:
    case CP_ACCESS_TRAP:
        break;
    case CP_ACCESS_TRAP_UNCATEGORIZED:
        env->exception.syndrome = syn_uncategorized();
        break;
    default:
        g_assert_not_reached();