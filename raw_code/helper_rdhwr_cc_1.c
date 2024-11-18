target_ulong helper_rdhwr_cc(CPUMIPSState *env)

{

    check_hwrena(env, 2);

#ifdef CONFIG_USER_ONLY

    return env->CP0_Count;

#else

    return (int32_t)cpu_mips_get_count(env);

#endif

}
