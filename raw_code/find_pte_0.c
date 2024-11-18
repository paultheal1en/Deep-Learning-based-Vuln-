static always_inline int find_pte (CPUState *env, mmu_ctx_t *ctx,

                                   int h, int rw)

{

#if defined(TARGET_PPC64)

    if (env->mmu_model == POWERPC_MMU_64B)

        return find_pte64(ctx, h, rw);

#endif



    return find_pte32(ctx, h, rw);

}
