static int cpu_pre_load(void *opaque)

{

    CPUState *env = opaque;



    cpu_synchronize_state(env);

    return 0;

}
