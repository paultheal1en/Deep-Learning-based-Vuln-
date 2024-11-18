int cpu_load(QEMUFile *f, void *opaque, int version_id)

{

    CPUState *env = opaque;

    int i;

    uint32_t tmp;



    if (version_id != 5)

        return -EINVAL;

    for(i = 0; i < 8; i++)

        qemu_get_betls(f, &env->gregs[i]);

    qemu_get_be32s(f, &env->nwindows);

    for(i = 0; i < env->nwindows * 16; i++)

        qemu_get_betls(f, &env->regbase[i]);



    /* FPU */

    for(i = 0; i < TARGET_FPREGS; i++) {

        union {

            float32 f;

            uint32_t i;

        } u;

        u.i = qemu_get_be32(f);

        env->fpr[i] = u.f;

    }



    qemu_get_betls(f, &env->pc);

    qemu_get_betls(f, &env->npc);

    qemu_get_betls(f, &env->y);

    tmp = qemu_get_be32(f);

    env->cwp = 0; /* needed to ensure that the wrapping registers are

                     correctly updated */

    PUT_PSR(env, tmp);

    qemu_get_betls(f, &env->fsr);

    qemu_get_betls(f, &env->tbr);

    tmp = qemu_get_be32(f);

    env->interrupt_index = tmp;

    qemu_get_be32s(f, &env->pil_in);

#ifndef TARGET_SPARC64

    qemu_get_be32s(f, &env->wim);

    /* MMU */

    for (i = 0; i < 32; i++)

        qemu_get_be32s(f, &env->mmuregs[i]);

#else

    qemu_get_be64s(f, &env->lsu);

    for (i = 0; i < 16; i++) {

        qemu_get_be64s(f, &env->immuregs[i]);

        qemu_get_be64s(f, &env->dmmuregs[i]);

    }

    for (i = 0; i < 64; i++) {

        qemu_get_be64s(f, &env->itlb[i].tag);

        qemu_get_be64s(f, &env->itlb[i].tte);

        qemu_get_be64s(f, &env->dtlb[i].tag);

        qemu_get_be64s(f, &env->dtlb[i].tte);

    }

    qemu_get_be32s(f, &env->mmu_version);

    for (i = 0; i < MAXTL_MAX; i++) {

        qemu_get_be64s(f, &env->ts[i].tpc);

        qemu_get_be64s(f, &env->ts[i].tnpc);

        qemu_get_be64s(f, &env->ts[i].tstate);

        qemu_get_be32s(f, &env->ts[i].tt);

    }

    qemu_get_be32s(f, &env->xcc);

    qemu_get_be32s(f, &env->asi);

    qemu_get_be32s(f, &env->pstate);

    qemu_get_be32s(f, &env->tl);

    env->tsptr = &env->ts[env->tl & MAXTL_MASK];

    qemu_get_be32s(f, &env->cansave);

    qemu_get_be32s(f, &env->canrestore);

    qemu_get_be32s(f, &env->otherwin);

    qemu_get_be32s(f, &env->wstate);

    qemu_get_be32s(f, &env->cleanwin);

    for (i = 0; i < 8; i++)

        qemu_get_be64s(f, &env->agregs[i]);

    for (i = 0; i < 8; i++)

        qemu_get_be64s(f, &env->bgregs[i]);

    for (i = 0; i < 8; i++)

        qemu_get_be64s(f, &env->igregs[i]);

    for (i = 0; i < 8; i++)

        qemu_get_be64s(f, &env->mgregs[i]);

    qemu_get_be64s(f, &env->fprs);

    qemu_get_be64s(f, &env->tick_cmpr);

    qemu_get_be64s(f, &env->stick_cmpr);

    qemu_get_ptimer(f, env->tick);

    qemu_get_ptimer(f, env->stick);

    qemu_get_be64s(f, &env->gsr);

    qemu_get_be32s(f, &env->gl);

    qemu_get_be64s(f, &env->hpstate);

    for (i = 0; i < MAXTL_MAX; i++)

        qemu_get_be64s(f, &env->htstate[i]);

    qemu_get_be64s(f, &env->hintp);

    qemu_get_be64s(f, &env->htba);

    qemu_get_be64s(f, &env->hver);

    qemu_get_be64s(f, &env->hstick_cmpr);

    qemu_get_be64s(f, &env->ssr);

    qemu_get_ptimer(f, env->hstick);

#endif

    tlb_flush(env, 1);

    return 0;

}
