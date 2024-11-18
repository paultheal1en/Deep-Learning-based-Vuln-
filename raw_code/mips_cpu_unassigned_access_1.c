void mips_cpu_unassigned_access(CPUState *cs, hwaddr addr,
                                bool is_write, bool is_exec, int unused,
                                unsigned size)
{
    MIPSCPU *cpu = MIPS_CPU(cs);
    CPUMIPSState *env = &cpu->env;
    if (is_exec) {
        helper_raise_exception(env, EXCP_IBE);
    } else {
        helper_raise_exception(env, EXCP_DBE);