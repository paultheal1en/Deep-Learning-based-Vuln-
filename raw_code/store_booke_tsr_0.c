void store_booke_tsr (CPUState *env, target_ulong val)

{

    LOG_TB("%s: val " TARGET_FMT_lx "\n", __func__, val);

    env->spr[SPR_40x_TSR] &= ~(val & 0xFC000000);

    if (val & 0x80000000)

        ppc_set_irq(env, PPC_INTERRUPT_PIT, 0);

}
