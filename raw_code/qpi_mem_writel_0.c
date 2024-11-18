static void qpi_mem_writel(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    CPUState *env;



    env = cpu_single_env;

    if (!env)

        return;

    env->eflags = (env->eflags & ~(IF_MASK | IOPL_MASK)) | 

        (val & (IF_MASK | IOPL_MASK));

}
