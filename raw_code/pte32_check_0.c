static int pte32_check (mmu_ctx_t *ctx,

                        target_ulong pte0, target_ulong pte1, int h, int rw)

{

    return _pte_check(ctx, 0, pte0, pte1, h, rw);

}
