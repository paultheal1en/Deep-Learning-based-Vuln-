static void gdb_breakpoint_remove_all(CPUState *env)

{

    cpu_breakpoint_remove_all(env, BP_GDB);

#ifndef CONFIG_USER_ONLY

    cpu_watchpoint_remove_all(env, BP_GDB);

#endif

}
