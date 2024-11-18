static void cpu_debug_handler(CPUState *env)

{

    gdb_set_stop_cpu(env);

    qemu_system_debug_request();

}
