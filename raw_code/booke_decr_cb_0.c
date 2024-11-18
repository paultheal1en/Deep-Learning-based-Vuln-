static void booke_decr_cb(void *opaque)

{

    PowerPCCPU *cpu = opaque;

    CPUPPCState *env = &cpu->env;



    env->spr[SPR_BOOKE_TSR] |= TSR_DIS;

    booke_update_irq(cpu);



    if (env->spr[SPR_BOOKE_TCR] & TCR_ARE) {

        /* Auto Reload */

        cpu_ppc_store_decr(env, env->spr[SPR_BOOKE_DECAR]);

    }

}
