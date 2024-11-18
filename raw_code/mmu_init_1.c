static void mmu_init (CPUMIPSState *env, const mips_def_t *def)

{

    env->tlb = qemu_mallocz(sizeof(CPUMIPSTLBContext));



    switch (def->mmu_type) {

        case MMU_TYPE_NONE:

            no_mmu_init(env, def);

            break;

        case MMU_TYPE_R4000:

            r4k_mmu_init(env, def);

            break;

        case MMU_TYPE_FMT:

            fixed_mmu_init(env, def);

            break;

        case MMU_TYPE_R3000:

        case MMU_TYPE_R6000:

        case MMU_TYPE_R8000:

        default:

            cpu_abort(env, "MMU type not supported\n");

    }

    env->CP0_Random = env->tlb->nb_tlb - 1;

    env->tlb->tlb_in_use = env->tlb->nb_tlb;

}
