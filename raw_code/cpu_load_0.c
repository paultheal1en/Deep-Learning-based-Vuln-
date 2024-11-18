int cpu_load(QEMUFile *f, void *opaque, int version_id)

{

	CPUCRISState *env = opaque;

    int i;

    int s;

    int mmu;



    for (i = 0; i < 16; i++)

        env->regs[i] = qemu_get_be32(f);

    for (i = 0; i < 16; i++)

        env->pregs[i] = qemu_get_be32(f);



    env->pc = qemu_get_be32(f);

    env->ksp = qemu_get_be32(f);



    env->dslot = qemu_get_be32(f);

    env->btaken = qemu_get_be32(f);

    env->btarget = qemu_get_be32(f);



    env->cc_op = qemu_get_be32(f);

    env->cc_mask = qemu_get_be32(f);

    env->cc_dest = qemu_get_be32(f);

    env->cc_src = qemu_get_be32(f);

    env->cc_result = qemu_get_be32(f);

    env->cc_size = qemu_get_be32(f);

    env->cc_x = qemu_get_be32(f);



    for (s = 0; s < 4; i++) {

        for (i = 0; i < 16; i++)

            env->sregs[s][i] = qemu_get_be32(f);

    }



    env->mmu_rand_lfsr = qemu_get_be32(f);

    for (mmu = 0; mmu < 2; mmu++) {

        for (s = 0; s < 4; i++) {

            for (i = 0; i < 16; i++) {

                env->tlbsets[mmu][s][i].lo = qemu_get_be32(f);

                env->tlbsets[mmu][s][i].hi = qemu_get_be32(f);

            }

        }

    }



    return 0;

}
