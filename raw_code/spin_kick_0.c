static void spin_kick(void *data)

{

    SpinKick *kick = data;

    CPUState *cpu = CPU(kick->cpu);

    CPUPPCState *env = &kick->cpu->env;

    SpinInfo *curspin = kick->spin;

    hwaddr map_size = 64 * 1024 * 1024;

    hwaddr map_start;



    cpu_synchronize_state(cpu);

    stl_p(&curspin->pir, env->spr[SPR_PIR]);

    env->nip = ldq_p(&curspin->addr) & (map_size - 1);

    env->gpr[3] = ldq_p(&curspin->r3);

    env->gpr[4] = 0;

    env->gpr[5] = 0;

    env->gpr[6] = 0;

    env->gpr[7] = map_size;

    env->gpr[8] = 0;

    env->gpr[9] = 0;



    map_start = ldq_p(&curspin->addr) & ~(map_size - 1);

    mmubooke_create_initial_mapping(env, 0, map_start, map_size);



    cpu->halted = 0;

    cpu->exception_index = -1;

    cpu->stopped = false;

    qemu_cpu_kick(cpu);

}
