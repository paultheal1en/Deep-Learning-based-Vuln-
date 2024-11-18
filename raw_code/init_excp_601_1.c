static void init_excp_601 (CPUPPCState *env)

{

#if !defined(CONFIG_USER_ONLY)

    env->excp_vectors[POWERPC_EXCP_RESET]    = 0x00000100;

    env->excp_vectors[POWERPC_EXCP_MCHECK]   = 0x00000200;

    env->excp_vectors[POWERPC_EXCP_DSI]      = 0x00000300;

    env->excp_vectors[POWERPC_EXCP_ISI]      = 0x00000400;

    env->excp_vectors[POWERPC_EXCP_EXTERNAL] = 0x00000500;

    env->excp_vectors[POWERPC_EXCP_ALIGN]    = 0x00000600;

    env->excp_vectors[POWERPC_EXCP_PROGRAM]  = 0x00000700;

    env->excp_vectors[POWERPC_EXCP_FPU]      = 0x00000800;

    env->excp_vectors[POWERPC_EXCP_DECR]     = 0x00000900;

    env->excp_vectors[POWERPC_EXCP_IO]       = 0x00000A00;

    env->excp_vectors[POWERPC_EXCP_SYSCALL]  = 0x00000C00;

    env->excp_vectors[POWERPC_EXCP_RUNM]     = 0x00002000;

    env->excp_prefix = 0xFFF00000;

    /* Hardware reset vector */

    env->hreset_vector = 0x00000100UL;

#endif

}
