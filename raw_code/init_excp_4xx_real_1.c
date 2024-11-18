static void init_excp_4xx_real (CPUPPCState *env)

{

#if !defined(CONFIG_USER_ONLY)

    env->excp_vectors[POWERPC_EXCP_CRITICAL] = 0x00000100;

    env->excp_vectors[POWERPC_EXCP_MCHECK]   = 0x00000200;

    env->excp_vectors[POWERPC_EXCP_EXTERNAL] = 0x00000500;

    env->excp_vectors[POWERPC_EXCP_ALIGN]    = 0x00000600;

    env->excp_vectors[POWERPC_EXCP_PROGRAM]  = 0x00000700;

    env->excp_vectors[POWERPC_EXCP_SYSCALL]  = 0x00000C00;

    env->excp_vectors[POWERPC_EXCP_PIT]      = 0x00001000;

    env->excp_vectors[POWERPC_EXCP_FIT]      = 0x00001010;

    env->excp_vectors[POWERPC_EXCP_WDT]      = 0x00001020;

    env->excp_vectors[POWERPC_EXCP_DEBUG]    = 0x00002000;

    env->excp_prefix = 0x00000000;

    env->ivor_mask = 0x0000FFF0;

    env->ivpr_mask = 0xFFFF0000;

    /* Hardware reset vector */

    env->hreset_vector = 0xFFFFFFFCUL;

#endif

}
